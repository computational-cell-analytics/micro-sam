import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from bioimage_cpp.utils import Blocking

import micro_sam.v2.batched_inference as batched_inference
from micro_sam.v2.batched_inference import (
    _compute_auto_batch_sizes,
    _decode_tiled_3d_embeddings,
    _decode_tiled_3d_slice,
    _decode_volume_embeddings,
    _run_batched_pipeline,
    _run_decoder_jobs,
    _select_throughput_batch_size,
)
from micro_sam.v2.prompt_based_segmentation import TiledPromptableSegmentation3D


class ArrayWithAttrs:
    def __init__(self, data, **attrs):
        self.data = np.asarray(data)
        self.attrs = attrs
        self.shape = self.data.shape

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, index):
        return self.data[index]


class Group(dict):
    def __init__(self, *args, **attrs):
        super().__init__(*args)
        self.attrs = attrs


class FakeEncoder(torch.nn.Module):
    def __init__(self, img_size=8):
        super().__init__()
        self.img_size = img_size


class FakeUNETR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(int(x.shape[0]))
        features = torch.stack(
            [self.encoder(x[:, :, index])[0] for index in range(x.shape[2])],
            dim=2,
        )
        prediction = features.mean(dim=1, keepdim=True)
        prediction = F.interpolate(
            prediction,
            size=(x.shape[2], *x.shape[-2:]),
            mode="trilinear",
            align_corners=False,
        )
        return prediction.repeat(1, 4, 1, 1, 1) * self.scale


class TestBatchedPipeline(unittest.TestCase):
    def test_pipeline_batches_and_writes_all_jobs(self):
        outputs = {}
        progress = []
        progress_threads = []

        def predict(model, items, device):
            self.assertEqual(device, torch.device("cpu"))
            return [value + model for value in items]

        def update_progress(update):
            progress.append(update)
            progress_threads.append(threading.get_ident())

        _run_batched_pipeline(
            jobs=range(7),
            model_devices=[(3, torch.device("cpu"))],
            batch_sizes=[3],
            load_fn=lambda value: 2 * value,
            predict_fn=predict,
            write_fn=outputs.__setitem__,
            num_prefetch_workers=2,
            update_progress=update_progress,
        )

        self.assertEqual(outputs, {index: 2 * index + 3 for index in range(7)})
        self.assertEqual(sum(progress), 7)
        self.assertEqual(set(progress_threads), {threading.get_ident()})

    def test_pipeline_writes_with_multiple_workers(self):
        outputs = {}
        writer_threads = set()
        lock = threading.Lock()

        def predict(model, items, device):
            return [value + model for value in items]

        def write(job, prediction):
            # Slow writes so the queue actually spreads over the writers.
            time.sleep(0.01)
            with lock:
                outputs[job] = prediction
                writer_threads.add(threading.get_ident())

        _run_batched_pipeline(
            jobs=range(16),
            model_devices=[(3, torch.device("cpu"))],
            batch_sizes=[2],
            load_fn=lambda value: 2 * value,
            predict_fn=predict,
            write_fn=write,
            num_prefetch_workers=2,
            num_write_workers=3,
        )

        self.assertEqual(outputs, {index: 2 * index + 3 for index in range(16)})
        self.assertGreater(len(writer_threads), 1)
        self.assertLessEqual(len(writer_threads), 3)
        self.assertNotIn(threading.get_ident(), writer_threads)

    def test_pipeline_does_not_deadlock_when_prediction_fails(self):
        # Regression: the sentinels were sent with a blocking put. A consumer that failed while the
        # input queue was full left nobody to drain it, so the producer (and the join) hung forever.
        def predict(model, items, device):
            time.sleep(0.3)  # let the producers reach their sentinel put with a full queue
            raise RuntimeError("synthetic prediction failure")

        errors = []

        def run():
            try:
                _run_batched_pipeline(
                    jobs=range(3),
                    model_devices=[(3, torch.device("cpu"))],
                    batch_sizes=[1],
                    load_fn=lambda value: value,
                    predict_fn=predict,
                    write_fn=lambda job, prediction: None,
                    num_prefetch_workers=3,
                )
            except Exception as exc:  # noqa
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=30)

        self.assertFalse(thread.is_alive(), "the pipeline deadlocked after a worker failure")
        self.assertEqual([str(error) for error in errors], ["synthetic prediction failure"])

    def test_pipeline_does_not_deadlock_when_a_write_fails(self):
        # The symmetric case: the consumers' sentinels go to the writers, which may already have died.
        def write(job, prediction):
            time.sleep(0.02)
            raise RuntimeError("synthetic write failure")

        errors = []

        def run():
            try:
                _run_batched_pipeline(
                    jobs=range(12),
                    model_devices=[(1, torch.device("cpu"))],
                    batch_sizes=[1],
                    load_fn=lambda value: value,
                    predict_fn=lambda model, items, device: list(items),
                    write_fn=write,
                    num_prefetch_workers=3,
                    num_write_workers=4,
                )
            except Exception as exc:  # noqa
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=30)

        self.assertFalse(thread.is_alive(), "the pipeline deadlocked after a write failure")
        self.assertEqual([str(error) for error in errors], ["synthetic write failure"])

    def test_pipeline_retries_ooming_batches(self):
        outputs = {}
        attempted_batch_sizes = []

        def predict(model, items, device):
            attempted_batch_sizes.append(len(items))
            if len(items) > 2:
                raise torch.cuda.OutOfMemoryError("synthetic test OOM")
            return [value + model for value in items]

        with self.assertWarnsRegex(RuntimeWarning, "retrying with smaller batches"):
            _run_batched_pipeline(
                jobs=range(7),
                model_devices=[(3, torch.device("cpu"))],
                batch_sizes=[4],
                load_fn=lambda value: 2 * value,
                predict_fn=predict,
                write_fn=outputs.__setitem__,
                num_prefetch_workers=2,
            )

        self.assertEqual(outputs, {index: 2 * index + 3 for index in range(7)})
        self.assertEqual(attempted_batch_sizes[0], 4)
        self.assertTrue(all(batch_size <= 2 for batch_size in attempted_batch_sizes[1:]))

    def test_pipeline_retries_singleton_after_cache_release(self):
        outputs = {}
        attempts = 0

        def predict(model, items, device):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise torch.cuda.OutOfMemoryError("synthetic fragmented-cache OOM")
            return [value + model for value in items]

        _run_batched_pipeline(
            jobs=[0], model_devices=[(3, torch.device("cpu"))], batch_sizes=[1],
            load_fn=lambda value: 2 * value, predict_fn=predict, write_fn=outputs.__setitem__,
        )

        self.assertEqual(outputs, {0: 3})
        self.assertEqual(attempts, 2)


class TestAutomaticBatchSizing(unittest.TestCase):
    def test_requires_material_throughput_improvement(self):
        self.assertEqual(
            _select_throughput_batch_size([(1, 10.0), (2, 10.5)]),
            1,
        )
        self.assertEqual(
            _select_throughput_batch_size([(1, 10.0), (2, 11.5)]),
            2,
        )

    @mock.patch(
        "micro_sam.v2.batched_inference._measure_batch_throughput",
        side_effect=[10.0, 10.5, 10.6],
    )
    def test_stops_after_two_non_improving_candidates(self, measure):
        model = torch.nn.Linear(1, 1)

        batch_sizes = _compute_auto_batch_sizes(
            model_devices=[(model, torch.device("cuda:0"))],
            n_jobs=64,
            patch_shape=(8, 8),
            in_channels=3,
            prediction_function=lambda this_model, inputs: this_model(inputs),
        )

        self.assertEqual(batch_sizes, [1])
        self.assertEqual(
            [call.kwargs["batch_size"] for call in measure.call_args_list],
            [1, 2, 4],
        )


class TestResolveDevices(unittest.TestCase):
    def test_fans_out_over_all_visible_cuda_devices(self):
        model = torch.nn.Linear(1, 1)
        with mock.patch.object(batched_inference, "_model_device", return_value=torch.device("cuda", 0)), \
                mock.patch.object(torch.cuda, "device_count", return_value=2), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            devices = batched_inference._resolve_devices(model, None)
        self.assertEqual(devices, [torch.device("cuda", 0), torch.device("cuda", 1)])

    def test_single_cuda_device_when_only_one_visible(self):
        model = torch.nn.Linear(1, 1)
        with mock.patch.object(batched_inference, "_model_device", return_value=torch.device("cuda", 0)), \
                mock.patch.object(torch.cuda, "device_count", return_value=1), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            devices = batched_inference._resolve_devices(model, None)
        self.assertEqual(devices, [torch.device("cuda", 0)])

    def test_cpu_model_stays_single_device(self):
        model = torch.nn.Linear(1, 1)
        self.assertEqual(batched_inference._resolve_devices(model, None), [torch.device("cpu")])

    def test_explicit_devices_are_used(self):
        model = torch.nn.Linear(1, 1)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            devices = batched_inference._resolve_devices(model, ["cuda:0", "cuda:1"])
        self.assertEqual(devices, [torch.device("cuda", 0), torch.device("cuda", 1)])

    def test_rejects_empty_and_duplicate_devices(self):
        model = torch.nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            batched_inference._resolve_devices(model, [])
        with self.assertRaises(ValueError):
            batched_inference._resolve_devices(model, ["cpu", "cpu"])


class TestPrepareModels(unittest.TestCase):
    def test_reuses_original_for_source_device_and_leaves_it_in_place(self):
        model = torch.nn.Linear(2, 2)
        source = batched_inference._model_device(model)
        pairs = batched_inference._prepare_models(model, [source])
        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0][0], model)
        self.assertEqual(batched_inference._model_device(model), source)

    def test_replicas_are_deep_copied_off_the_source_device(self):
        # F4: the deepcopy must run while the model is on CPU, so a second copy never lands on the
        # source device. We spoof a two-CUDA setup without hardware by stubbing device moves.
        from copy import deepcopy as real_deepcopy

        model = torch.nn.Linear(2, 2)
        source = torch.device("cuda", 0)
        events = []

        def fake_to(self, device, *args, **kwargs):
            events.append(("to", str(device)))
            return self

        def fake_deepcopy(obj):
            events.append(("deepcopy", None))
            return real_deepcopy(obj)

        with mock.patch.object(batched_inference, "_model_device", return_value=source), \
                mock.patch.object(torch.nn.Module, "to", fake_to), \
                mock.patch.object(batched_inference, "deepcopy", fake_deepcopy):
            batched_inference._prepare_models(model, [source, torch.device("cuda", 1)])

        # Move to CPU first, then deepcopy there, and restore the original to the source device.
        self.assertEqual(events[0], ("to", "cpu"))
        self.assertEqual(events[1], ("deepcopy", None))
        self.assertIn(("to", "cuda:0"), events)


class TestBatchedDecoder(unittest.TestCase):
    def test_volume_decoder_batches_z_blocks(self):
        features = np.arange(8 * 2 * 2 * 2, dtype="float32").reshape(8, 2, 2, 2)
        model = FakeUNETR()
        progress = []

        output = _decode_volume_embeddings(
            model,
            {"features": features, "original_size": (8, 8)},
            z_block=2,
            z_halo=0,
            batch_size=2,
            num_prefetch_workers=2,
            pbar_update=progress.append,
        )

        self.assertEqual(output.shape, (4, 8, 8, 8))
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual(sum(progress), 8)

    def test_tiled_volume_decoder_batches_tiles_and_z_blocks(self):
        features = Group(
            {
                str(tile_id): ArrayWithAttrs(
                    np.full((4, 1, 2, 2, 2), tile_id + 1, dtype="float32"),
                    original_size=(4, 4),
                )
                for tile_id in range(4)
            },
            shape=(4, 8, 8),
            tile_shape=(4, 4),
            halo=(0, 0),
        )
        model = FakeUNETR()
        progress = []

        output = _decode_tiled_3d_embeddings(
            model,
            {"features": features},
            z_block=2,
            z_halo=0,
            batch_size=2,
            num_prefetch_workers=2,
            pbar_update=progress.append,
        )

        self.assertEqual(output.shape, (4, 4, 8, 8))
        self.assertTrue(np.allclose(output[:, :, :4, :4], 1))
        self.assertTrue(np.allclose(output[:, :, :4, 4:], 2))
        self.assertTrue(np.allclose(output[:, :, 4:, :4], 3))
        self.assertTrue(np.allclose(output[:, :, 4:, 4:], 4))
        self.assertEqual(model.batch_sizes, [2, 2, 2, 2])
        self.assertEqual(sum(progress), 16)

    def test_largest_decoder_shape_runs_first(self):
        jobs = [
            {"source": np.ones((2, 2, 2, 2), dtype="float32"), "original_size": (8, 8), "name": "small"},
            {"source": np.ones((4, 2, 2, 2), dtype="float32"), "original_size": (8, 8), "name": "large"},
        ]
        write_order = []

        _run_decoder_jobs(
            FakeUNETR(), jobs, lambda job, prediction: write_order.append(job["name"]),
            batch_size=1, devices="cpu", num_prefetch_workers=1,
        )

        self.assertEqual(write_order, ["large", "small"])


class TestDecoderEncoderRestoration(unittest.TestCase):
    def test_encoder_is_restored_when_job_inspection_fails(self):
        # Regression: the encoder was swapped for the decoder-only replica before the job shapes were
        # inspected, so a malformed job left the caller's model with the placeholder encoder.
        model = FakeUNETR()
        encoder = model.encoder
        malformed = [{"source": np.ones((2,) * 6, dtype="float32"), "original_size": (8, 8)}]

        with self.assertRaises(ValueError):
            _run_decoder_jobs(model, malformed, lambda job, prediction: None, batch_size=1, devices="cpu")
        self.assertIs(model.encoder, encoder)

    def test_encoder_is_restored_when_batch_size_is_invalid(self):
        model = FakeUNETR()
        encoder = model.encoder
        jobs = [{"source": np.ones((2, 2, 2, 2), dtype="float32"), "original_size": (8, 8)}]

        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            _run_decoder_jobs(model, jobs, lambda job, prediction: None, batch_size=0, devices="cpu")
        self.assertIs(model.encoder, encoder)

    def test_encoder_is_restored_when_a_write_fails(self):
        model = FakeUNETR()
        encoder = model.encoder
        jobs = [{"source": np.ones((2, 2, 2, 2), dtype="float32"), "original_size": (8, 8)}]

        def write(job, prediction):
            raise RuntimeError("synthetic write failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic write failure"):
            _run_decoder_jobs(model, jobs, write, batch_size=1, devices="cpu")
        self.assertIs(model.encoder, encoder)


class FakeInteractiveTile:
    def __init__(self, value):
        self.value = value

    def predict(self, update_progress=None, **kwargs):
        if update_progress is not None:
            update_progress(2)
        return np.full((2, 4, 4), self.value, dtype="uint64")


class TestInteractiveTileScheduling(unittest.TestCase):
    def test_active_tile_columns_are_stitched_after_device_jobs(self):
        segmenter = TiledPromptableSegmentation3D.__new__(TiledPromptableSegmentation3D)
        segmenter.shape = (2, 8, 8)
        segmenter.halo = (0, 0)
        segmenter.tiling = Blocking([0, 0], [8, 8], [4, 4])
        segmenter._predictor_devices = [(None, torch.device("cpu")), (None, torch.device("cpu"))]
        segmenter._tile_workers = {}
        segmenter._segmenters = {
            tile_id: FakeInteractiveTile(tile_id + 1)
            for tile_id in range(4)
        }
        progress = []
        progress_threads = []

        def update_progress(value):
            progress.append(value)
            progress_threads.append(threading.get_ident())

        output = segmenter.predict(update_progress=update_progress)

        self.assertTrue(np.all(output[:, :4, :4] == 1))
        self.assertTrue(np.all(output[:, :4, 4:] == 2))
        self.assertTrue(np.all(output[:, 4:, :4] == 3))
        self.assertTrue(np.all(output[:, 4:, 4:] == 4))
        self.assertEqual(sum(progress), 8)
        self.assertEqual(set(progress_threads), {threading.get_ident()})


class TestZBlockValidation(unittest.TestCase):
    def _tiled_embeddings(self):
        features = Group(
            {
                "0": ArrayWithAttrs(np.zeros((4, 1, 2, 2, 2), dtype="float32"), original_size=(4, 4)),
            },
            shape=(4, 4, 4),
            tile_shape=(4, 4),
            halo=(0, 0),
        )
        return {"features": features}

    def test_tiled_decoder_rejects_invalid_z_blocking(self):
        # Regression: a negative z_block made the job range empty, so an all-zero prediction was
        # returned without ever running the decoder.
        model = FakeUNETR()
        for z_block, z_halo in [(-1, 0), (0, 0), (2, -1)]:
            with self.assertRaisesRegex(ValueError, "z_block must be positive"):
                _decode_tiled_3d_embeddings(
                    model, self._tiled_embeddings(), z_block=z_block, z_halo=z_halo, batch_size=1,
                )

    def test_volume_decoder_rejects_invalid_z_blocking(self):
        model = FakeUNETR()
        embeddings = {"features": np.zeros((4, 2, 2, 2), dtype="float32"), "original_size": (8, 8)}
        for z_block, z_halo in [(-1, 0), (0, 0), (2, -1)]:
            with self.assertRaisesRegex(ValueError, "z_block must be positive"):
                _decode_volume_embeddings(model, embeddings, z_block=z_block, z_halo=z_halo, batch_size=1)

    def test_tiled_slice_decoder_rejects_out_of_range_index(self):
        # A negative index would decode from the end, an index past the volume nothing at all.
        model = FakeUNETR()
        for index in (-1, 4, 10):
            with self.assertRaisesRegex(ValueError, "slice index must be in"):
                _decode_tiled_3d_slice(model, self._tiled_embeddings(), index=index, batch_size=1)

    def test_tiled_slice_decoder_accepts_valid_index(self):
        model = FakeUNETR()
        output = _decode_tiled_3d_slice(model, self._tiled_embeddings(), index=3, batch_size=1)
        self.assertEqual(output.shape, (4, 4, 4))


class TestTileDeviceAffinity(unittest.TestCase):
    def _segmenter(self, n_devices):
        segmenter = TiledPromptableSegmentation3D.__new__(TiledPromptableSegmentation3D)
        segmenter._predictor_devices = [(None, torch.device("cpu"))] * n_devices
        segmenter._tile_workers = {}
        return segmenter

    def test_sparse_tiles_are_spread_over_devices(self):
        # Regression: 'tile_id % n_devices' mapped the tiles 0, 2, 4, 6 to the same device.
        segmenter = self._segmenter(2)
        segmenter._run_tile_jobs([0, 2, 4, 6], lambda tile_id: tile_id)
        self.assertEqual([segmenter._worker_id(tile_id) for tile_id in (0, 2, 4, 6)], [0, 1, 0, 1])

    def test_tile_affinity_is_kept_across_jobs(self):
        # The tile state lives on one device, so its assignment must not change between jobs.
        segmenter = self._segmenter(3)
        segmenter._run_tile_jobs([3, 6], lambda tile_id: tile_id)
        assignment = dict(segmenter._tile_workers)
        self.assertEqual(sorted(assignment.values()), [0, 1])

        segmenter._run_tile_jobs([6, 3, 9], lambda tile_id: tile_id)
        for tile_id, worker_id in assignment.items():
            self.assertEqual(segmenter._worker_id(tile_id), worker_id)
        self.assertEqual(segmenter._worker_id(9), 2)


if __name__ == "__main__":
    unittest.main()
