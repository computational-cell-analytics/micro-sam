import platform
import unittest
from unittest import mock

import numpy as np
import pytest

import micro_sam.util as util
import micro_sam.v2.util as v2_util
import micro_sam.sam_annotator._widgets as widgets
from micro_sam.v2.util import VRAM_BATCH_SIZES, recommend_batch_size


class _FakeSpinBox:
    """Stands in for the QSpinBox, so the widget logic is testable without a display."""

    def __init__(self, value=1):
        self._value = value
        self.signals_blocked = False

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value

    def blockSignals(self, block):
        self.signals_blocked = block


class _FakeContainer:
    def __init__(self):
        self.visible = None

    def setVisible(self, visible):
        self.visible = visible


def _make_widget(model_type="hvit_t_cells", device="cuda", ndim=3, tiling="no"):
    """An EmbeddingWidget with only the attributes the batch-size logic touches.

    Allocated without '__init__' so no Qt widgets are built, which keeps this runnable headless.
    """
    widget = widgets.EmbeddingWidget.__new__(widgets.EmbeddingWidget)
    widget.model_type = model_type
    widget.device = device
    widget.batch_size = 1
    widget._batch_size_is_auto = True
    widget.batch_size_param = _FakeSpinBox()
    widget._batch_size_widget = _FakeContainer()
    widget.tiling = tiling
    widget._ndim_override = lambda: None
    widget._selected_image_ndim = lambda: ndim
    return widget


class TestRecommendedBatchSize(unittest.TestCase):
    def test_sam2_model_uses_the_table(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(widget._recommended_batch_size(), VRAM_BATCH_SIZES[80]["hvit_t"])

    def test_smaller_card_gets_a_smaller_batch(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=9.64):
            self.assertEqual(widget._recommended_batch_size(), VRAM_BATCH_SIZES[8]["hvit_t"])

    def test_heavier_backbone_gets_its_own_entry(self):
        widget = _make_widget(model_type="hvit_l")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=4.0):
            self.assertEqual(widget._recommended_batch_size(), VRAM_BATCH_SIZES[4]["hvit_l"])

    def test_sam1_model_stays_at_one(self):
        # The table is calibrated for the SAM2 encoders only.
        widget = _make_widget(model_type="vit_b_lm")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(widget._recommended_batch_size(), 1)

    def test_cpu_stays_at_one(self):
        widget = _make_widget(device="cpu")
        self.assertEqual(widget._recommended_batch_size(), 1)

    def test_auto_device_is_resolved(self):
        widget = _make_widget(device="auto")
        with mock.patch.object(widgets.util, "_get_default_device", return_value="cuda"), \
                mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(widget._recommended_batch_size(), VRAM_BATCH_SIZES[80]["hvit_t"])


class TestRefreshBatchSize(unittest.TestCase):
    def test_refresh_applies_the_recommendation(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        expected = VRAM_BATCH_SIZES[80]["hvit_t"]
        self.assertEqual(widget.batch_size, expected)
        self.assertEqual(widget.batch_size_param.value(), expected)

    def test_refresh_does_not_re_enter_the_edit_handler(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        # Signals are blocked around the programmatic write, and released afterwards.
        self.assertTrue(widget._batch_size_is_auto)
        self.assertFalse(widget.batch_size_param.signals_blocked)

    def test_a_user_edit_is_kept_across_a_device_switch(self):
        widget = _make_widget()
        widget._on_batch_size_edited(7)
        widget.batch_size = 7
        widget.batch_size_param.setValue(7)

        widget.device = "cpu"
        widget._refresh_batch_size()
        self.assertEqual(widget.batch_size, 7)

    def test_switching_device_updates_the_value(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[80]["hvit_t"])

        widget.device = "cpu"
        widget._refresh_batch_size()
        self.assertEqual(widget.batch_size, 1)

    def test_switching_model_updates_the_value(self):
        widget = _make_widget(model_type="hvit_t")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=4.0):
            widget._refresh_batch_size()
            self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[4]["hvit_t"])

            widget.model_type = "hvit_l"
            widget._refresh_batch_size()
            self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[4]["hvit_l"])

    def test_visibility_follows_the_device(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertTrue(widget._batch_size_widget.visible)

        widget.device = "cpu"
        widget._refresh_batch_size()
        self.assertFalse(widget._batch_size_widget.visible)

    def test_hidden_on_a_cpu_only_machine(self):
        # 'auto' resolving to the CPU must hide the field too, not just an explicit CPU selection.
        widget = _make_widget(device="auto")
        with mock.patch.object(widgets.util, "_get_default_device", return_value="cpu"):
            widget._refresh_batch_size()
        self.assertFalse(widget._batch_size_widget.visible)
        self.assertEqual(widget.batch_size, 1)

    def test_hidden_for_vfm_encoders(self):
        widget = _make_widget(model_type="vit_b_dinov2")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertFalse(widget._batch_size_widget.visible)
        self.assertEqual(widget.batch_size, 1)

    def test_hidden_for_a_non_tiled_2d_image(self):
        # One image without tiling is a single encoder call, so there is nothing to batch over.
        widget = _make_widget(ndim=2, tiling="no")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertFalse(widget._batch_size_widget.visible)
        self.assertEqual(widget._effective_batch_size(), 1)

    def test_shown_for_a_tiled_2d_image(self):
        widget = _make_widget(ndim=2, tiling="yes")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertTrue(widget._batch_size_widget.visible)
        self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[80]["hvit_t"])

    def test_shown_for_a_volume_without_tiling(self):
        widget = _make_widget(ndim=3, tiling="no")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertTrue(widget._batch_size_widget.visible)
        self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[80]["hvit_t"])

    def test_shown_while_no_image_is_selected(self):
        # Dimensionality is unknown before an image is picked; the field appears once it is known.
        widget = _make_widget(ndim=None, tiling="no")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertTrue(widget._batch_size_widget.visible)

    def test_forcing_2d_on_a_volume_hides_it(self):
        widget = _make_widget(ndim=3, tiling="no")
        widget._ndim_override = lambda: 2
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
        self.assertFalse(widget._batch_size_widget.visible)


class TestEffectiveBatchSize(unittest.TestCase):
    """The displayed value is a preview: it is read before the model is loaded and, for 'auto', from
    the default device only. So while the user has not edited it, the backend chooses instead."""

    def test_the_recommendation_is_not_forwarded(self):
        widget = _make_widget()
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
            self.assertEqual(widget.batch_size, VRAM_BATCH_SIZES[80]["hvit_t"])
            self.assertIsNone(widget._effective_batch_size())

    def test_a_value_the_user_typed_is_forwarded(self):
        widget = _make_widget()
        widget._on_batch_size_edited(7)
        widget.batch_size = 7
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            self.assertEqual(widget._effective_batch_size(), 7)

    def test_a_model_outside_the_table_is_forwarded(self):
        # SAM1 has no per-device lookup in the backend, and its embedding function needs a number.
        widget = _make_widget(model_type="vit_b_lm")
        with mock.patch.object(v2_util, "_free_vram_gib", return_value=92.6):
            widget._refresh_batch_size()
            self.assertEqual(widget._effective_batch_size(), 1)

    def test_without_an_effect_it_is_one(self):
        widget = _make_widget(device="cpu")
        self.assertEqual(widget._effective_batch_size(), 1)


# The tests above drive the methods on a stand-in, so they cannot see whether the widget wires them
# up correctly. These build the real widget: 'model_type' used to be resolved only when a dropdown
# changed or the run button was pressed, so a freshly opened annotator showed a batch size of one.
@pytest.mark.gui
def test_widget_resolves_the_model_type_at_construction(qapp):
    widget = widgets.EmbeddingWidget()
    assert widget.model_type, "model_type must be set before any dropdown changes"


@pytest.mark.gui
def test_widget_starts_at_the_recommended_batch_size(qapp):
    # Compared against the table directly rather than the widget's own helper, which would agree
    # with it trivially when the model type is missing and both fall back to one.
    widget = widgets.EmbeddingWidget()
    if not widget._batch_size_has_effect():
        pytest.skip("batching has no effect without a GPU")

    device = util._get_default_device() if widget.device == "auto" else widget.device
    expected = recommend_batch_size(widget.model_type, device)
    assert expected > 1, "a GPU should be given a batch larger than one"
    assert widget.batch_size == expected
    assert widget.batch_size_param.value() == expected


@pytest.mark.gui
def test_widget_follows_the_device_dropdown(qapp):
    widget = widgets.EmbeddingWidget()
    if not widget._batch_size_has_effect():
        pytest.skip("batching has no effect without a GPU")
    on_gpu = widget.batch_size
    assert on_gpu > 1

    widget.device_dropdown.setCurrentText("cpu")
    assert widget.batch_size == 1
    assert not widget._batch_size_has_effect()

    widget.device_dropdown.setCurrentText("auto")
    assert widget.batch_size == on_gpu


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="GUI test does not work on Windows.")
def test_widget_tracks_dimensionality_and_tiling(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    widget = widgets.EmbeddingWidget()
    if not widget._batch_size_has_effect():
        pytest.skip("batching has no effect without a GPU")

    viewer.add_image(np.zeros((64, 64), dtype="uint8"), name="flat")
    widget.image_selection.reset_choices()
    widget.image_selection.value = viewer.layers["flat"]
    widget._set_default_tiling()
    assert not widget._batch_size_has_effect(), "a non-tiled 2d image has nothing to batch"
    assert widget._effective_batch_size() == 1

    widget.tiling_dropdown.setCurrentText("yes")
    assert widget._batch_size_has_effect(), "tiles are what the batch spans"
    assert widget.batch_size > 1

    viewer.add_image(np.zeros((8, 64, 64), dtype="uint8"), name="volume")
    widget.image_selection.reset_choices()
    widget.image_selection.value = viewer.layers["volume"]
    widget.tiling_dropdown.setCurrentText("no")
    assert widget._batch_size_has_effect(), "z slices are what the batch spans"
    assert widget.batch_size > 1


@pytest.mark.gui
def test_widget_keeps_a_value_the_user_typed(qapp):
    widget = widgets.EmbeddingWidget()
    widget.batch_size_param.setValue(7)
    assert not widget._batch_size_is_auto

    widget.device_dropdown.setCurrentText("cpu")
    widget.device_dropdown.setCurrentText("auto")
    assert widget.batch_size == 7


if __name__ == "__main__":
    unittest.main()
