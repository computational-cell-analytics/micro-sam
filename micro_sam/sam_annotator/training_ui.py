import warnings
# from pathlib import Path
# from typing import Optional

import torch
import torch_em
from napari.qt.threading import thread_worker
from qtpy import QtWidgets
from torch.utils.data import random_split

import micro_sam.util as util
import micro_sam.sam_annotator._widgets as widgets
from micro_sam.training import default_sam_dataset, train_sam_for_setting, SETTINGS


def _find_best_setting():
    if torch.cuda.is_available():
        # TODO
        # can we check the GPU type and use it to match the setting?
        return "rtx5000"
    else:
        return "CPU"


class TrainingWidget(widgets._WidgetBase):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Create the UI: the general options.
        self.layout().addLayout(self._create_options())

        # Add the settings (collapsible).
        self.layout().addWidget(self._create_settings_widget())

        # Add the run button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Start Training")
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

    def _create_options(self):
        # TODO add raw_path, label_path, raw_key and label_key.
        # we need _add_string_param and _add_path_param for this.
        # wait for Luca's PR on this.
        self.raw_path = None
        self.raw_key = None

        self.label_path = None
        self.label_key = None

        self.setting = _find_best_setting()
        self._add_choice_param("setting", self.setting, list(SETTINGS.keys()))

        self.train_instance_segmentation = True
        self._add_boolean_param("train_instance_segmentation", self.train_instance_segmentation)

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # TODO use CPU instead of MPS on MAC because training with MPS is slower!
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()
        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options)
        setting_values.layout().addLayout(layout)

        self.patch_x, self.patch_y = 0, 0
        self.patch_x_param, self.patch_y_param, layout = self._add_shape_param(
            ("patch_x", "patch_y"), (self.patch_x, self.patch_y), min_val=0, max_val=2048
        )
        setting_values.layout().addLayout(layout)

        # TODO: the other optional params, also include stuff for val data!
        self.name = None
        self.initial_model_name = None
        self.checkpoint_path = None

        settings = widgets._make_collapsible(setting_values, title="Advanced")
        return settings

    def _get_loaders(self):
        batch_size = 1
        num_workers = 1 if str(self.device) == "cpu" else 4

        patch_shape = (self.patch_shape_x, self.patch_shape_y)
        # TODO use the values from optional params instead.
        raw_path_val, label_path_val = None, None
        if raw_path_val is None:
            dataset = default_sam_dataset(
                str(self.raw_path), self.raw_key, str(self.label_path), self.label_key,
                patch_shape=patch_shape, with_segmentation_decoder=self.train_instance_segmentation,
            )
            # TODO better heuristic for the split?
            train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - 1, 1])
        else:
            if label_path_val is None:
                raise ValueError
            # TODO
            train_dataset, val_dataset = "", ""

        train_loader = torch_em.segmentation.get_data_loader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = torch_em.segmentation.get_data_loader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        return train_loader, val_loader

    def _get_model_type(self):
        # Consolidate initial model name, the checkpoint path and the model type according to the settings.
        if self.self.initial_model_name is None or self.initial_model_name in ("None", ""):
            model_type = SETTINGS[self.setting]["model_type"]
        else:
            model_type = self.initial_model_name[:5]
            if model_type != SETTINGS[self.setting]["model_type"]:
                warnings.warn(
                    f"You have changed the model type for your chosen setting {self.setting} "
                    f"from {SETTINGS[self.setting]['model_type']} to {model_type}. "
                    "The training may be very slow or not work at all."
                )
        assert model_type is not None
        return model_type

    def __call__(self):
        # Set up progress bar and signals for using it within a threadworker.
        pbar, pbar_signals = widgets._create_pbar_for_threadworker()

        model_type = self._get_model_type()
        if self.checkpoint_path is None:
            model_registry = util.models()
            checkpoint_path = model_registry.fetch(model_type)
        else:
            checkpoint_path = self.checkpoint_path

        @thread_worker()
        def run_training():
            train_loader, val_loader = self._get_loaders()
            # TODO enable passing pbar callbacks to torch_em
            train_sam_for_setting(
                name=self.name, setting=self.setting,
                train_loader=train_loader, val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                with_segmentation_decoder=self.train_instance_segmentation,
                model_type=model_type, device=self.device
            )
            # TODO export the model

        worker = run_training()
        # Note: this is how we can handle the worker when it's done.
        # We can use this e.g. to add an indicator that the embeddings are computed or not.
        # worker.returned.connect(lambda _: print("Embeddings for", self.model_type, "have been computed."))
        worker.start()
        return worker
