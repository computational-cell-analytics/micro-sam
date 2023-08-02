"""The main GUI for starting annotation tools.
"""

import os
import magicgui
import numpy as np
import zarr
from magicgui.widgets import Container, Label, LineEdit, SpinBox, ComboBox
from magicgui.application import use_app
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ..util import load_image_data, get_model_names
from .annotator_2d import annotator_2d
from .annotator_3d import annotator_3d
from .image_series_annotator import image_folder_annotator
from .annotator_tracking import annotator_tracking

config_dict = {}
"""@private"""
main_widget = None
"""@private"""


def show_error(msg):
    """@private"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setText(msg)
    msg_box.setWindowTitle("Error")
    msg_box.exec()


def file_is_hirarchical(path_s):
    """@private"""
    if isinstance(path_s, list):
        return all([file_is_hirarchical(path) for path in path_s])
    else:
        return os.path.splitext(path_s)[1] in [".hdf5", ".h5", "n5", ".zarr"]


def _set_embeddings_file_attributes(embeddings_path: str, cb_model, re_tile_x, re_tile_y):
    f = zarr.open(embeddings_path, "a")
    if "tile_shape" in f.attrs:
        if f.attrs["tile_shape"] is None:
            re_tile_x.value = 0
            re_tile_y.value = 0
        else:
            re_tile_x.value = f.attrs["tile_shape"][0]
            re_tile_y.value = f.attrs["tile_shape"][1]
    if "model_type" in f.attrs:
        cb_model.value = f.attrs["model_type"]


@magicgui.magicgui(call_button="2d annotator", labels=False)
def _on_2d():
    global config_dict
    sub_widget = None
    config_dict["args"] = {}
    args = config_dict["args"]

    le_file_key_img = LineEdit(value="*", label="File key input")
    le_file_key_segm = LineEdit(value="*", label="File key segmentation")
    re_halo_x = SpinBox(value=0, max=10000, label="Halo x")
    re_halo_y = SpinBox(value=0, max=10000, label="Halo y")
    re_tile_x = SpinBox(value=0, max=10000, label="Tile x")
    re_tile_y = SpinBox(value=0, max=10000, label="Tile y")
    cb_model = ComboBox(value="vit_h_lm", choices=get_model_names(), label="Model Type")

    @magicgui.magicgui(call_button="Select image", labels=False)
    def on_select_image():
        try:
            path = QFileDialog.getOpenFileName(None, 'Open file', '.', "Image files (*)")[0]
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            key = le_file_key_img.value if file_is_hirarchical(path) else None
            args["raw"] = load_image_data(path, key=key)
        except Exception as e:
            show_error(str(e))

    lbl_opt = Label(value="Optional:")

    @magicgui.magicgui(call_button="Select embeddings file", labels=False)
    def on_select_embed():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path) or os.path.splitext(path)[1] != ".zarr":
                show_error("Precompute embeddings file does not exist or has wrong file extension.")
                return
            args["embedding_path"] = path
            _set_embeddings_file_attributes(path, cb_model, re_tile_x, re_tile_y)
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select segmentation result", labels=False)
    def on_select_segm():
        try:
            path = QFileDialog.getOpenFileName(None, 'Open file', '.', "Image files (*)")[0]
            if path == "":
                return
            if not os.path.exists(path):
                return
            key = le_file_key_segm.value if file_is_hirarchical(path) else None
            args["segmentation_result"] = load_image_data(path, key=key)
        except Exception as e:
            show_error(str(e))

    pb_img_sel = Container(widgets=[on_select_image], layout="horizontal", labels=False)
    pb_embed_sel = Container(widgets=[on_select_embed], layout="horizontal", labels=False)
    pb_seg_segm = Container(widgets=[on_select_segm], layout="horizontal", labels=False)

    @magicgui.magicgui(call_button="2d annotator", labels=False)
    def on_start():
        try:
            if "raw" not in args.keys():
                show_error("Input file was not selected")
                return

            tile_shape = (int(re_tile_x.value), int(re_tile_y.value))
            if tile_shape[0] > 0 and tile_shape[1] > 0:
                if "embedding_path" not in args.keys():
                    show_error("If tiling is used, embeddings file must be set")
                    return
                args["tile_shape"] = tile_shape

            halo = (int(re_halo_x.value), int(re_halo_y.value))
            if halo[0] > 0 or tile_shape[1] > 0:
                args["halo"] = halo
            args["model_type"] = cb_model.value
            sub_widget.close()
            config_dict["workflow"] = "2d"
        except Exception as e:
            show_error(str(e))

    sub_widget = Container(widgets=[Container(widgets=[on_start], layout="horizontal", labels=False),
                                    pb_img_sel, lbl_opt, pb_embed_sel, pb_seg_segm, le_file_key_img, le_file_key_segm,
                                    cb_model, re_tile_x, re_tile_y, re_halo_x, re_halo_y])
    sub_widget.root_native_widget.setWindowTitle("Segment Anything for Microscopy")
    main_widget.close()
    sub_widget.show()


@magicgui.magicgui(call_button="3d annotator", labels=False)
def _on_3d():
    global config_dict
    config_dict["args"] = {}
    args = config_dict["args"]
    le_file_key_img = LineEdit(value="*", label="File key input")
    le_file_key_segm = LineEdit(value="*", label="File key segmentation")

    re_halo_x = SpinBox(value=0, max=10000, label="Halo x")
    re_halo_y = SpinBox(value=0, max=10000, label="Halo y")
    re_tile_x = SpinBox(value=0, max=10000, label="Tile x")
    re_tile_y = SpinBox(value=0, max=10000, label="Tile y")
    cb_model = ComboBox(value="vit_h_lm", choices=get_model_names(), label="Model Type")

    @magicgui.magicgui(call_button="Select images", labels=False)
    def on_select_image():
        try:
            paths = QFileDialog.getOpenFileNames(None, 'Open file', '.', "Image files (*)")[0]
            if not all([os.path.exists(path) for path in paths]):
                show_error("Could not find all requested files.")
                return
            key = le_file_key_img.value if file_is_hirarchical(paths) else None
            imgs = np.stack([load_image_data(path, key=key) for path in paths])
            args["raw"] = np.squeeze(imgs) if imgs.shape[0] == 1 else imgs
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select image directory", labels=False)
    def on_select_image_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".")
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["raw"] = load_image_data(path, key=le_file_key_img.value)
        except Exception as e:
            show_error(str(e))

    lbl_opt = Label(value="Optional:")

    @magicgui.magicgui(call_button="Select embeddings file", labels=False)
    def on_select_embed():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path) or os.path.splitext(path)[1] != ".zarr":
                show_error("Precompute embeddings file does not exist or has wrong file extension.")
                return
            args["embedding_path"] = path
            _set_embeddings_file_attributes(path, cb_model, re_tile_x, re_tile_y)
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select segmentation result", labels=False)
    def on_select_segm():
        try:
            paths = QFileDialog.getOpenFileNames(None, 'Open file', '.', "Image files (*)")[0]
            if not all([os.path.exists(path) for path in paths]):
                show_error("Could not find all requested files.")
                return
            key = le_file_key_segm.value if file_is_hirarchical(paths) else None
            imgs = np.stack([load_image_data(path, key=key) for path in paths])
            args["segmentation_result"] = np.squeeze(imgs) if imgs.shape[0] == 1 else imgs
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select segmentation result directory", labels=False)
    def on_select_segm_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".")
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["segmentation_result"] = load_image_data(path, key=le_file_key_segm.value)
        except Exception as e:
            show_error(str(e))

    pb_img_sel = Container(widgets=[on_select_image, on_select_image_dir], layout="horizontal", labels=False)
    pb_embed_sel = Container(widgets=[on_select_embed], layout="horizontal", labels=False)
    pb_seg_segm = Container(widgets=[on_select_segm, on_select_segm_dir], layout="horizontal", labels=False)

    @magicgui.magicgui(call_button="3d annotator", labels=False)
    def on_start():
        try:
            if "raw" not in args.keys():
                show_error("Input file was not selected")
                return

            tile_shape = (int(re_tile_x.value), int(re_tile_y.value))
            if tile_shape[0] > 0 and tile_shape[1] > 0:
                if "embedding_path" not in args.keys():
                    show_error("If tiling is used, embeddings file must be set")
                    return
                args["tile_shape"] = tile_shape

            halo = (int(re_halo_x.value), int(re_halo_y.value))
            if halo[0] > 0 or tile_shape[1] > 0:
                args["halo"] = halo
            args["model_type"] = cb_model.value
            sub_widget.close()
            config_dict["workflow"] = "3d"
        except Exception as e:
            show_error(str(e))

    sub_widget = Container(widgets=[Container(widgets=[on_start], layout="horizontal", labels=False),
                                    pb_img_sel, lbl_opt, pb_embed_sel, pb_seg_segm, le_file_key_img, le_file_key_segm,
                                    cb_model, re_tile_x, re_tile_y, re_halo_x, re_halo_y])
    sub_widget.root_native_widget.setWindowTitle("Segment Anything for Microscopy")
    main_widget.close()
    sub_widget.show()


@magicgui.magicgui(call_button="Image series annotator", labels=False)
def _on_series():
    global config_dict
    config_dict["args"] = {}
    args = config_dict["args"]

    re_halo_x = SpinBox(value=0, max=10000, label="Halo x")
    re_halo_y = SpinBox(value=0, max=10000, label="Halo y")
    re_tile_x = SpinBox(value=0, max=10000, label="Tile x")
    re_tile_y = SpinBox(value=0, max=10000, label="Tile y")
    cb_model = ComboBox(value="vit_h_lm", choices=get_model_names(), label="Model Type")

    @magicgui.magicgui(call_button="Select input directory", labels=False)
    def on_select_input_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["input_folder"] = path
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select output directory", labels=False)
    def on_select_output_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["output_folder"] = path
        except Exception as e:
            show_error(str(e))

    lbl_opt = Label(value="Optional:")

    @magicgui.magicgui(call_button="Select embeddings file", labels=False)
    def on_select_embed():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path) or os.path.splitext(path)[1] != ".zarr":
                show_error("Precompute embeddings file does not exist or has wrong file extension.")
                return
            args["embedding_path"] = path
            _set_embeddings_file_attributes(path, cb_model, re_tile_x, re_tile_y)
        except Exception as e:
            show_error(str(e))

    pb_input_sel = Container(widgets=[on_select_input_dir], layout="horizontal", labels=False)
    pb_output_sel = Container(widgets=[on_select_output_dir], layout="horizontal", labels=False)
    pb_embed_sel = Container(widgets=[on_select_embed], layout="horizontal", labels=False)

    @magicgui.magicgui(call_button="Image series annotator", labels=False)
    def on_start():
        try:
            if "output_folder" not in args.keys():
                show_error("Output folder was not selected.")
                return
            if "input_folder" not in args.keys():
                show_error("Input folder was not selected.")
                return
            tile_shape = (int(re_tile_x.value), int(re_tile_y.value))
            if tile_shape[0] > 0 and tile_shape[1] > 0:
                if "embedding_path" not in args.keys():
                    show_error("If tiling is used, embeddings file must be set")
                    return
                args["tile_shape"] = tile_shape

            halo = (int(re_halo_x.value), int(re_halo_y.value))
            if halo[0] > 0 or tile_shape[1] > 0:
                args["halo"] = halo
            args["model_type"] = cb_model.value
            sub_widget.close()
            config_dict["workflow"] = "series"
        except Exception as e:
            show_error(str(e))

    sub_widget = Container(widgets=[Container(widgets=[on_start], layout="horizontal", labels=False),
                                    pb_input_sel, pb_output_sel, lbl_opt, pb_embed_sel, cb_model, re_tile_x,
                                    re_tile_y, re_halo_x, re_halo_y])
    sub_widget.root_native_widget.setWindowTitle("Segment Anything for Microscopy")
    main_widget.close()
    sub_widget.show()


@magicgui.magicgui(call_button="Tracking annotator", labels=False)
def _on_tracking():
    global config_dict
    config_dict["args"] = {}
    args = config_dict["args"]
    le_file_key_img = LineEdit(value="*", label="File key input")
    le_file_key_segm = LineEdit(value="*", label="File key segmentation")

    re_halo_x = SpinBox(value=0, max=10000, label="Halo x")
    re_halo_y = SpinBox(value=0, max=10000, label="Halo y")
    re_tile_x = SpinBox(value=0, max=10000, label="Tile x")
    re_tile_y = SpinBox(value=0, max=10000, label="Tile y")
    cb_model = ComboBox(value="vit_h_lm", choices=get_model_names(), label="Model Type")

    @magicgui.magicgui(call_button="Select images", labels=False)
    def on_select_image():
        try:
            paths = QFileDialog.getOpenFileNames(None, 'Open file', '.', "Image files (*)")[0]
            if not all([os.path.exists(path) for path in paths]):
                show_error("Could not find all requested files.")
                return
            key = le_file_key_img.value if file_is_hirarchical(paths) else None
            imgs = np.stack([load_image_data(path, key=key) for path in paths])
            args["raw"] = np.squeeze(imgs) if imgs.shape[0] == 1 else imgs
        except Exception as e:
            show_error(str(e))

    lbl_opt = Label(value="Optional:")

    @magicgui.magicgui(call_button="Select image directory", labels=False)
    def on_select_image_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".")
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["raw"] = load_image_data(path, key=le_file_key_img.value)
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select embeddings file", labels=False)
    def on_select_embed():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".", QFileDialog.ShowDirsOnly)
            if path == "":
                return
            if not os.path.exists(path) or os.path.splitext(path)[1] != ".zarr":
                show_error("Precompute embeddings file does not exist or has wrong file extension.")
                return
            args["embedding_path"] = path
            _set_embeddings_file_attributes(path, cb_model, re_tile_x, re_tile_y)
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select tracking result", labels=False)
    def on_select_results():
        try:
            paths = QFileDialog.getOpenFileNames(None, 'Open file', '.', "Image files (*)")[0]
            if not all([os.path.exists(path) for path in paths]):
                show_error("Could not find all requested files.")
                return
            key = le_file_key_segm.value if file_is_hirarchical(paths) else None
            imgs = np.stack([load_image_data(path, key=key) for path in paths])
            args["tracking_result"] = np.squeeze(imgs) if imgs.shape[0] == 1 else imgs
        except Exception as e:
            show_error(str(e))

    @magicgui.magicgui(call_button="Select tracking result directory", labels=False)
    def on_select_result_dir():
        try:
            path = QFileDialog.getExistingDirectory(None, "Open a folder", ".")
            if path == "":
                return
            if not os.path.exists(path):
                show_error("Could not find all requested files.")
                return
            args["tracking_result"] = load_image_data(path, key=le_file_key_segm.value)
        except Exception as e:
            show_error(str(e))

    pb_img_sel = Container(widgets=[on_select_image, on_select_image_dir], layout="horizontal", labels=False)
    pb_embed_sel = Container(widgets=[on_select_embed], layout="horizontal", labels=False)
    pb_seg_sel = Container(widgets=[on_select_results, on_select_result_dir], layout="horizontal", labels=False)

    @magicgui.magicgui(call_button="Tracking annotator", labels=False)
    def on_start():
        try:
            if "raw" not in args.keys():
                show_error("Input file was not selected")
                return

            tile_shape = (int(re_tile_x.value), int(re_tile_y.value))
            if tile_shape[0] > 0 and tile_shape[1] > 0:
                if "embedding_path" not in args.keys():
                    show_error("If tiling is used, embeddings file must be set")
                    return
                args["tile_shape"] = tile_shape

            halo = (int(re_halo_x.value), int(re_halo_y.value))
            if halo[0] > 0 or tile_shape[1] > 0:
                args["halo"] = halo
            args["model_type"] = cb_model.value
            sub_widget.close()
            config_dict["workflow"] = "tracking"
        except Exception as e:
            show_error(str(e))

    sub_widget = Container(widgets=[Container(widgets=[on_start], layout="horizontal", labels=False),
                                    pb_img_sel, lbl_opt, pb_embed_sel, pb_seg_sel, le_file_key_img, le_file_key_segm,
                                    cb_model, re_tile_x, re_tile_y, re_halo_x, re_halo_y])
    sub_widget.root_native_widget.setWindowTitle("Segment Anything for Microscopy")
    main_widget.close()
    sub_widget.show()


def annotator():
    """Start the main micro_sam GUI.

    From this GUI you can select which annotation tool you want to use and then
    select the parameters for the tool.
    """
    global main_widget, config_dict
    config_dict["workflow"] = ""
    sub_container1 = Container(widgets=[_on_2d, _on_series], labels=False)
    sub_container2 = Container(widgets=[_on_3d, _on_tracking], labels=False)
    sub_container3 = Container(widgets=[sub_container1, sub_container2], layout="horizontal", labels=False)
    main_widget = Container(widgets=[Label(value="Segment Anything for Microscopy"), sub_container3], labels=False)
    main_widget.root_native_widget.setWindowTitle("Segment Anything for Microscopy")
    main_widget.show(run=True)

    if config_dict["workflow"] == "2d":
        use_app().quit()
        annotator_2d(show_embeddings=False, **config_dict["args"])
    elif config_dict["workflow"] == "3d":
        use_app().quit()
        annotator_3d(show_embeddings=False, **config_dict["args"])
    elif config_dict["workflow"] == "series":
        use_app().quit()
        image_folder_annotator(**config_dict["args"])
    elif config_dict["workflow"] == "tracking":
        use_app().quit()
        annotator_tracking(show_embeddings=False, **config_dict["args"])
    else:
        show_error("No workflow selected.")
        use_app().quit()


def main():
    """@private"""
    annotator()
