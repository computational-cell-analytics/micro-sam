import os
import warnings

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
        self._create_options()

        # Add the settings (collapsible).
        self.layout().addWidget(self._create_settings())

        # Add the run button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Start Training")
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

    def _create_options(self):
        self.raw_path = None
        _, layout = self._add_path_param(
            "raw_path", self.raw_path, "both", placeholder="Image data ..."
        )
        self.layout().addLayout(layout)

        self.raw_key = None
        _, layout = self._add_string_param(
            "raw_key", self.raw_key, placeholder="Image data key ...",
        )
        self.layout().addLayout(layout)

        self.label_path = None
        _, layout = self._add_path_param(
            "label_path", self.label_path, "both", placeholder="Label data ..."
        )
        self.layout().addLayout(layout)

        self.label_key = None
        _, layout = self._add_string_param(
            "label_key", self.label_key, placeholder="Label data key ...",
        )
        self.layout().addLayout(layout)

        self.setting = _find_best_setting()
        self.setting_dropdown, layout = self._add_choice_param("setting", self.setting, list(SETTINGS.keys()))
        self.layout().addLayout(layout)

        self.with_segmentation_decoder = True
        self.layout().addWidget(self._add_boolean_param(
            "with_segmentation_decoder", self.with_segmentation_decoder
        ))

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # TODO use CPU instead of MPS on MAC because training with MPS is slower!
        # Device and patch shape settings.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()
        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options)
        setting_values.layout().addLayout(layout)

        self.patch_x, self.patch_y = 512, 512
        self.patch_x_param, self.patch_y_param, layout = self._add_shape_param(
            ("patch_x", "patch_y"), (self.patch_x, self.patch_y), min_val=0, max_val=2048
        )
        setting_values.layout().addLayout(layout)

        # Paths for validation data.
        self.raw_path_val = None
        _, layout = self._add_path_param(
            "raw_path_val", self.raw_path_val, "both", placeholder="Image data for validation ..."
        )
        setting_values.layout().addLayout(layout)

        self.label_path_val = None
        _, layout = self._add_path_param(
            "label_path_val", self.label_path_val, "both", placeholder="Label data for validation ..."
        )
        setting_values.layout().addLayout(layout)

        # Name of the model to be trained and options to over-ride the initial model
        # on top of which the finetuning is run.
        self.name = "sam_model"
        self.name_param, layout = self._add_string_param("name", self.name)
        setting_values.layout().addLayout(layout)

        self.initial_model = None
        self.initial_model_param, layout = self._add_string_param("initial_model", self.initial_model)
        setting_values.layout().addLayout(layout)

        self.checkpoint = None
        self.checkpoint_param, layout = self._add_string_param("checkpoint", self.name)
        setting_values.layout().addLayout(layout)

        settings = widgets._make_collapsible(setting_values, title="Advanced")
        return settings

    def _get_loaders(self):
        batch_size = 1
        num_workers = 1 if str(self.device) == "cpu" else 4

        patch_shape = (self.patch_x, self.patch_y)
        dataset = default_sam_dataset(
            str(self.raw_path), self.raw_key, str(self.label_path), self.label_key,
            patch_shape=patch_shape, with_segmentation_decoder=self.with_segmentation_decoder,
        )

        raw_path_val, label_path_val = self.raw_path_val, self.label_path_val
        if raw_path_val is None:
            # TODO better heuristic for the split?
            train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - 1, 1])
        else:
            train_dataset = dataset
            val_dataset = default_sam_dataset(
                str(raw_path_val), self.raw_key, str(label_path_val), self.label_key,
                patch_shape=patch_shape, with_segmentation_decoder=self.with_segmentation_decoder,
            )

        train_loader = torch_em.segmentation.get_data_loader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = torch_em.segmentation.get_data_loader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        return train_loader, val_loader

    def _get_model_type(self):
        # Consolidate initial model name, the checkpoint path and the model type according to the settings.
        if self.initial_model is None or self.initial_model in ("None", ""):
            model_type = SETTINGS[self.setting]["model_type"]
        else:
            model_type = self.initial_model[:5]
            if model_type != SETTINGS[self.setting]["model_type"]:
                warnings.warn(
                    f"You have changed the model type for your chosen setting {self.setting} "
                    f"from {SETTINGS[self.setting]['model_type']} to {model_type}. "
                    "The training may be very slow or not work at all."
                )
        assert model_type is not None
        return model_type

    # Make sure that raw and label path have been passed.
    # If they haven't raise an error message.
    # (We could do a more extensive validation here, but for now keep it minimal.)
    def _validate_inputs(self):
        missing_raw = self.raw_path is None or not os.path.exists(self.raw_path)
        missing_label = self.label_path is None or not os.path.exists(self.label_path)
        if missing_raw or missing_label:
            msg = ""
            if missing_raw:
                msg += "The path to raw data is missing or does not exist. "
            if missing_label:
                msg += "The path to label data is missing or does not exist."
            return widgets._generate_message("error", msg)
        return False

    def __call__(self, skip_validate=False):
        if not skip_validate and self._validate_inputs():
            return

        # Set up progress bar and signals for using it within a threadworker.
        pbar, pbar_signals = widgets._create_pbar_for_threadworker()

        model_type = self._get_model_type()
        if self.checkpoint is None:
            model_registry = util.models()
            checkpoint_path = model_registry.fetch(model_type)
        else:
            checkpoint_path = self.checkpoint

        @thread_worker()
        def run_training():
            train_loader, val_loader = self._get_loaders()
            train_sam_for_setting(
                name=self.name, setting=self.setting,
                train_loader=train_loader, val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                with_segmentation_decoder=self.with_segmentation_decoder,
                model_type=model_type, device=self.device,
                pbar_signals=pbar_signals,
            )

            # TODO implement exporting the model to SAM (state dict) or bioimage.io format.

            pbar_signals.pbar_stop.emit()

        worker = run_training()
        worker.start()
        return worker
