import unittest
from unittest import mock

import micro_sam.v2.util as v2_util
from micro_sam.v2.util import (
    BAND_TOLERANCE,
    FALLBACK_BACKBONE,
    VRAM_BATCH_SIZES,
    _backbone_of,
    _band_for,
    recommend_batch_size,
)

# Peak reserved VRAM of one encoder call, as fixed + per_sample * batch (GiB) at 1024x1024.
ENCODER_COST = {
    "hvit_t": (0.82, 0.53),
    "hvit_s": (0.86, 0.53),
    "hvit_b": (1.08, 0.61),
    "hvit_l": (1.81, 0.78),
}


class TestTable(unittest.TestCase):
    def test_every_band_covers_every_backbone(self):
        backbones = set(VRAM_BATCH_SIZES[min(VRAM_BATCH_SIZES)])
        self.assertEqual(backbones, set(ENCODER_COST))
        for entry in VRAM_BATCH_SIZES.values():
            self.assertEqual(set(entry), backbones)

    def test_batch_sizes_are_positive(self):
        for entry in VRAM_BATCH_SIZES.values():
            for batch_size in entry.values():
                self.assertGreaterEqual(batch_size, 1)

    def test_batch_size_never_decreases_with_vram(self):
        for backbone in ENCODER_COST:
            values = [VRAM_BATCH_SIZES[band][backbone] for band in sorted(VRAM_BATCH_SIZES)]
            self.assertEqual(values, sorted(values))

    def test_heavier_backbones_never_get_larger_batches(self):
        ordered = sorted(ENCODER_COST, key=lambda name: ENCODER_COST[name][1])
        for band, entry in VRAM_BATCH_SIZES.items():
            values = [entry[backbone] for backbone in ordered]
            self.assertEqual(values, sorted(values, reverse=True), f"band {band}")

    def test_every_entry_fits_the_band_it_is_tabulated_at(self):
        for band, entry in VRAM_BATCH_SIZES.items():
            for backbone, batch_size in entry.items():
                fixed, per_sample = ENCODER_COST[backbone]
                predicted = fixed + per_sample * batch_size
                self.assertLessEqual(predicted, band * 0.8, f"{backbone} at {band} GiB")


class TestBands(unittest.TestCase):
    def test_device_maps_to_the_largest_band_it_reaches(self):
        self.assertEqual(_band_for(92.6), 80)
        self.assertEqual(_band_for(80.0), 80)
        self.assertEqual(_band_for(19.4), 16)

    def test_nominal_card_sizes_reach_their_own_band(self):
        # No card reports its nominal size as free, so each of these would otherwise fall a band
        # short: an 80 GB A100 has 79.25 GiB free and would be batched as a 48 GiB card.
        for nominal, free in [(8, 7.7), (10, 9.6), (12, 11.6), (16, 15.6), (24, 23.6),
                              (32, 31.6), (40, 39.1), (48, 47.4), (80, 79.25)]:
            self.assertEqual(_band_for(free), nominal, f"{nominal} GB card with {free} GiB free")

    def test_a_busy_card_drops_to_a_lower_band(self):
        # The slack must not let a card whose memory is genuinely in use claim a higher band.
        self.assertEqual(_band_for(60.0), 48)
        self.assertEqual(_band_for(20.0), 16)

    def test_every_entry_fits_the_least_free_vram_of_its_band(self):
        for band, entry in VRAM_BATCH_SIZES.items():
            for backbone, batch_size in entry.items():
                fixed, per_sample = ENCODER_COST[backbone]
                predicted = fixed + per_sample * batch_size
                self.assertLess(predicted, band * BAND_TOLERANCE, f"{backbone} at {band} GiB")

    def test_below_the_smallest_band_reaches_nothing(self):
        self.assertIsNone(_band_for(1.0))


class TestBackboneResolution(unittest.TestCase):
    def test_finetuned_name_maps_to_backbone(self):
        self.assertEqual(_backbone_of("hvit_t_cells"), "hvit_t")

    def test_plain_backbone_maps_to_itself(self):
        for backbone in ENCODER_COST:
            self.assertEqual(_backbone_of(backbone), backbone)

    def test_unknown_name_falls_back_conservatively(self):
        self.assertEqual(_backbone_of("vit_b_lm"), FALLBACK_BACKBONE)
        # The fallback must never be batched more aggressively than any other backbone.
        for entry in VRAM_BATCH_SIZES.values():
            self.assertEqual(entry[FALLBACK_BACKBONE], min(entry.values()))


class TestRecommendBatchSize(unittest.TestCase):
    def test_non_cuda_device_uses_batch_one(self):
        self.assertEqual(recommend_batch_size("hvit_t", "cpu"), 1)

    def test_n_jobs_caps_the_batch_size(self):
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(recommend_batch_size("hvit_t", "cuda", n_jobs=1), 1)
            self.assertEqual(recommend_batch_size("hvit_t", "cuda", n_jobs=0), 1)

    def test_uses_the_table_entry_for_the_band(self):
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(recommend_batch_size("hvit_t_cells", "cuda"), VRAM_BATCH_SIZES[80]["hvit_t"])

    def test_scarce_vram_gets_a_smaller_batch(self):
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=4.0):
            crowded = recommend_batch_size("hvit_t", "cuda")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            empty = recommend_batch_size("hvit_t", "cuda")
        self.assertLess(crowded, empty)

    def test_heavy_model_on_a_small_device(self):
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=4.0):
            self.assertEqual(recommend_batch_size("hvit_l", "cuda"), 1)

    def test_a_device_below_every_band_stays_at_one(self):
        # The smallest entry is calibrated for the smallest band. Applying it to a device that does
        # not reach that band would OOM and recover only through the backoff.
        for backbone in ENCODER_COST:
            with mock.patch.object(v2_util, "_free_vram_gib", return_value=2.0):
                self.assertEqual(recommend_batch_size(backbone, "cuda"), 1, backbone)


if __name__ == "__main__":
    unittest.main()
