import os
import warnings

from qtpy import QtWidgets
# from napari.qt.threading import thread_worker

import torch_em
from torch.utils.data import random_split

import micro_sam.util as util
import micro_sam.sam_annotator._widgets as widgets
from micro_sam.training.training import _find_best_configuration, _export_helper
from micro_sam.training import default_sam_dataset, train_sam_for_configuration, CONFIGURATIONS

from ._tooltips import get_tooltip


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
            "raw_path", self.raw_path, "both", placeholder="/path/to/images", title="Path to images",
            tooltip=get_tooltip("training", "raw_path")
        )
        self.layout().addLayout(layout)

        self.raw_key = None
        _, layout = self._add_string_param(
            "raw_key", self.raw_key, placeholder="e.g. \"*.tif\"", title="Image data key",
            tooltip=get_tooltip("training", "raw_key")
        )
        self.layout().addLayout(layout)

        self.label_path = None
        _, layout = self._add_path_param(
            "label_path", self.label_path, "both", placeholder="/path/to/labels", title="Path to labels",
            tooltip=get_tooltip("training", "label_path")
        )
        self.layout().addLayout(layout)

        self.label_key = None
        _, layout = self._add_string_param(
            "label_key", self.label_key, placeholder="e.g. \"*.tif\"", title="Label data key",
            tooltip=get_tooltip("training", "label_key")
        )
        self.layout().addLayout(layout)

        self.configuration = _find_best_configuration()
        self.setting_dropdown, layout = self._add_choice_param(
            "configuration", self.configuration, list(CONFIGURATIONS.keys()), title="Configuration",
            tooltip=get_tooltip("training", "configuration")
        )
        self.layout().addLayout(layout)

        self.with_segmentation_decoder = True
        self.layout().addWidget(self._add_boolean_param(
            "with_segmentation_decoder", self.with_segmentation_decoder, title="With segmentation decoder",
            tooltip=get_tooltip("training", "segmentation_decoder")
        ))

    def _get_model_size_options(self):
        # We store the actual model names mapped to UI labels.
        self.model_size_mapping = {}
        if self.model_family == "Natural Images (SAM)":
            self.model_size_options = list(self._model_size_map .values())
            self.model_size_mapping = {self._model_size_map[k]: f"vit_{k}" for k in self._model_size_map.keys()}
        else:
            model_suffix = self.supported_dropdown_maps[self.model_family]
            self.model_size_options = []

            for option in self.model_options:
                if option.endswith(model_suffix):
                    # Extract model size character on-the-fly.
                    key = next((k for k in self._model_size_map .keys() if f"vit_{k}" in option), None)
                    if key:
                        size_label = self._model_size_map[key]
                        self.model_size_options.append(size_label)
                        self.model_size_mapping[size_label] = option  # Store the actual model name.

        # We ensure an assorted order of model sizes ('tiny' to 'huge')
        self.model_size_options.sort(key=lambda x: ["tiny", "base", "large", "huge"].index(x))

    def _update_model_type(self):
        # Get currently selected model size (before clearing dropdown)
        current_selection = self.model_size_dropdown.currentText()
        self._get_model_size_options()  # Update model size options dynamically

        # NOTE: We need to prevent recursive updates for this step temporarily.
        self.model_size_dropdown.blockSignals(True)

        # Let's clear and recreate the dropdown.
        self.model_size_dropdown.clear()
        self.model_size_dropdown.addItems(self.model_size_options)

        # We restore the previous selection, if still valid.
        if current_selection in self.model_size_options:
            self.model_size = current_selection
        else:
            if self.model_size_options:  # Default to the first available model size
                self.model_size = self.model_size_options[0]

        # Let's map the selection to the correct model type (eg. "tiny" -> "vit_t")
        size_key = next(
            (k for k, v in self._model_size_map.items() if v == self.model_size), "b"
        )
        self.model_type = f"vit_{size_key}" + self.supported_dropdown_maps[self.model_family]

        self.model_size_dropdown.setCurrentText(self.model_size)  # Apply the selected text to the dropdown

        # We force a refresh for UI here.
        self.model_size_dropdown.update()

        # NOTE: And finally, we should re-enable signals again.
        self.model_size_dropdown.blockSignals(False)

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # TODO use CPU instead of MPS on MAC because training with MPS is slower!
        # Device and patch shape settings.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()
        self.device_dropdown, layout = self._add_choice_param(
            "device", self.device, device_options, title="Device", tooltip=get_tooltip("training", "device")
        )
        setting_values.layout().addLayout(layout)

        self.patch_x, self.patch_y = 512, 512
        self.patch_x_param, self.patch_y_param, layout = self._add_shape_param(
            ("patch_x", "patch_y"), (self.patch_x, self.patch_y), min_val=0, max_val=2048,
            tooltip=get_tooltip("training", "patch"), title=("Patch size x", "Patch size y")
        )
        setting_values.layout().addLayout(layout)

        # Paths for validation data.
        self.raw_path_val = None
        _, layout = self._add_path_param(
            "raw_path_val", self.raw_path_val, "both", placeholder="/path/to/images",
            title="Path to validation images", tooltip=get_tooltip("training", "raw_path_val")
        )
        setting_values.layout().addLayout(layout)

        self.label_path_val = None
        _, layout = self._add_path_param(
            "label_path_val", self.label_path_val, "both", placeholder="/path/to/images",
            title="Path to validation labels", tooltip=get_tooltip("training", "label_path_val")
        )
        setting_values.layout().addLayout(layout)

        # Name of the model to be trained and options to over-ride the initial model
        # on top of which the finetuning is run.
        self.name = "sam_model"
        self.name_param, layout = self._add_string_param(
            "name", self.name, title="Model name", tooltip=get_tooltip("training", "name")
        )
        setting_values.layout().addLayout(layout)

        # Add the model family and model size combination.

        # Create a UI with a list of support dropdown values and correspond them to suffixes.
        self.supported_dropdown_maps = {
            "Natural Images (SAM)": "",
            "Light Microscopy": "_lm",
            "Electron Microscopy": "_em_organelles",
            "Medical Imaging": "_medical_imaging",
            "Histopathology": "_histopathology",
        }

        self._model_size_map = {"t": "tiny", "b": "base", "l": "large", "h": "huge"}

        self._default_model_choice = "vit_b"  # NOTE: for finetuning, we set the default to "vit_b"
        # Let's set the literally default model choice depending on 'micro-sam'.
        self.model_family = {v: k for k, v in self.supported_dropdown_maps.items()}[self._default_model_choice[5:]]

        # NOTE: We stick to the base variant for each model family.
        # i.e. 'Natural Images (SAM)', 'Light Microscopy', 'Electron Microscopy', 'Medical_Imaging', 'Histopathology'.
        self.model_family_dropdown, layout = self._add_choice_param(
            "model_family", self.model_family, list(self.supported_dropdown_maps.keys()),
            title="Model Family", tooltip=get_tooltip("embedding", "model_family")
        )
        self.model_family_dropdown.currentTextChanged.connect(self._update_model_type)
        setting_values.layout().addLayout(layout)

        # Create UI for the model size.
        # This would combine with the chosen 'self.model_family' and depend on 'self._default_model_choice'.
        self.model_size = self._model_size_map[self._default_model_choice[4]]

        # Get all model options.
        self.model_options = list(util.models().urls.keys())
        # Filter out the decoders from the model list.
        self.model_options = [model for model in self.model_options if not model.endswith("decoder")]

        # Now, we get the available sizes per model family.
        self._get_model_size_options()

        self.model_size_dropdown, layout = self._add_choice_param(
            "model_size", self.model_size, self.model_size_options,
            title="Model Size", tooltip=get_tooltip("embedding", "model_size"),
        )
        self.model_size_dropdown.currentTextChanged.connect(self._update_model_type)
        setting_values.layout().addLayout(layout)

        self.checkpoint = None
        self.checkpoint_param, layout = self._add_string_param(
            "checkpoint", self.name, title="Checkpoint", tooltip=get_tooltip("training", "checkpoint")
        )
        setting_values.layout().addLayout(layout)

        self.output_path = None
        self.output_path_param, layout = self._add_string_param(
            "output_path", self.output_path, title="Output Path", tooltip=get_tooltip("training", "output_path")
        )
        setting_values.layout().addLayout(layout)

        self.n_epochs = 100
        self.n_epochs_param, layout = self._add_int_param(
            "n_epochs", self.n_epochs, title="Number of epochs", min_val=1, max_val=1000,
            tooltip=get_tooltip("training", "n_epochs"),
        )
        setting_values.layout().addLayout(layout)

        settings = widgets._make_collapsible(setting_values, title="Advanced Settings")
        return settings

    def _get_loaders(self):
        batch_size = 1
        num_workers = 1 if str(self.device) == "cpu" else 4

        patch_shape = (self.patch_x, self.patch_y)
        dataset = default_sam_dataset(
            raw_paths=str(self.raw_path),
            raw_key=self.raw_key,
            label_paths=str(self.label_path),
            label_key=self.label_key,
            patch_shape=patch_shape,
            with_segmentation_decoder=self.with_segmentation_decoder,
        )

        raw_path_val, label_path_val = self.raw_path_val, self.label_path_val
        if raw_path_val is None:
            # Use 10% of the dataset - at least one image - for validation.
            n_val = min(1, int(0.1 * len(dataset)))
            train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - n_val, n_val])
        else:
            train_dataset = dataset
            val_dataset = default_sam_dataset(
                raw_paths=str(raw_path_val),
                raw_key=self.raw_key,
                label_paths=str(label_path_val),
                label_key=self.label_key,
                patch_shape=patch_shape,
                with_segmentation_decoder=self.with_segmentation_decoder,
            )

        train_loader = torch_em.segmentation.get_data_loader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = torch_em.segmentation.get_data_loader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        return train_loader, val_loader

    def _get_model_type(self):
        # Let's get them all into one `model_type`.
        self.initial_model = "vit_" + self.model_size[0] + self.supported_dropdown_maps[self.model_family]

        # Consolidate initial model name, the checkpoint path and the model type according to the configuration.
        model_type = CONFIGURATIONS[self.configuration]["model_type"]
        if self.initial_model[:5] != model_type:
            warnings.warn(
                f"You have changed the model type for your chosen configuration {self.configuration} "
                f"from {CONFIGURATIONS[self.configuration]['model_type']} to {model_type}. "
                "The training may be extremely slow. Please be aware of your custom model choice."
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

            # For provided checkpoint path, we remove the displayed text on top of the drop-down menu.
            # NOTE: We prevent recursive updates for this step temporarily.
            self.model_family_dropdown.blockSignals(True)
            self.model_family_dropdown.setCurrentIndex(-1)  # This removes the displayed text.
            self.model_family_dropdown.update()
            # NOTE: And re-enable signals again.
            self.model_family_dropdown.blockSignals(False)

        # @thread_worker()
        def run_training():
            train_loader, val_loader = self._get_loaders()
            train_sam_for_configuration(
                name=self.name,
                configuration=self.configuration,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                with_segmentation_decoder=self.with_segmentation_decoder,
                model_type=model_type,
                device=self.device,
                n_epochs=self.n_epochs,
                pbar_signals=pbar_signals,
            )

            # The best checkpoint after training.
            export_checkpoint = os.path.join("checkpoints", self.name, "best.pt")
            assert os.path.exists(export_checkpoint), export_checkpoint

            output_path = _export_helper(
                "", self.name, self.output_path, model_type, self.with_segmentation_decoder, val_loader
            )
            pbar_signals.pbar_stop.emit()
            return output_path

        path = run_training()
        print(f"Training has finished. The trained model is saved at {path}.")
        # worker = run_training()
        # worker.returned.connect(lambda path: print(f"Training has finished. The trained model is saved at {path}."))
        # worker.start()
        # return worker
