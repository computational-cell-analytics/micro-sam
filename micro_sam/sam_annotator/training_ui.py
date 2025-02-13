import os
import warnings

from qtpy import QtWidgets
# from napari.qt.threading import thread_worker

import torch
from torch.utils.data import random_split

import torch_em

import micro_sam.util as util
import micro_sam.sam_annotator._widgets as widgets
from micro_sam.training.training import _find_best_configuration
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

        self.initial_model = None
        self.initial_model_param, layout = self._add_string_param(
            "initial_model", self.initial_model, title="Initial model", tooltip=get_tooltip("training", "initial_model")
        )
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
        # Consolidate initial model name, the checkpoint path and the model type according to the configuration.
        if self.initial_model is None or self.initial_model in ("None", ""):
            model_type = CONFIGURATIONS[self.configuration]["model_type"]
        else:
            model_type = self.initial_model[:5]
            if model_type != CONFIGURATIONS[self.configuration]["model_type"]:
                warnings.warn(
                    f"You have changed the model type for your chosen configuration {self.configuration} "
                    f"from {CONFIGURATIONS[self.configuration]['model_type']} to {model_type}. "
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

        # @thread_worker()
        def run_training():
            train_loader, val_loader = self._get_loaders()
            train_sam_for_configuration(
                name=self.name, configuration=self.configuration,
                train_loader=train_loader, val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                with_segmentation_decoder=self.with_segmentation_decoder,
                model_type=model_type, device=self.device,
                n_epochs=self.n_epochs, pbar_signals=pbar_signals,
            )

            # The best checkpoint after training.
            export_checkpoint = os.path.join("checkpoints", self.name, "best.pt")
            assert os.path.exists(export_checkpoint), export_checkpoint

            # Export the model if an output path was given.
            if self.output_path:

                # If the output path has a pytorch specific ending then
                # we just export the checkpoint.
                if os.path.splitext(self.output_path)[1] in (".pt", ".pth"):
                    util.export_custom_sam_model(
                        checkpoint_path=export_checkpoint, model_type=model_type, save_path=self.output_path,
                    )

                # Otherwise we export it as bioimage.io model.
                else:
                    from micro_sam.bioimageio import export_sam_model

                    # Load image and label image from the val loader.
                    with torch.no_grad():
                        image, label_image = next(iter(val_loader))
                        image, label_image = image.cpu().numpy().squeeze(), label_image.cpu().numpy().squeeze()

                    # Select the first channel of the label image if we have a channel axis.
                    # (This contains the labels.)
                    if label_image.ndim == 3:
                        label_image = label_image[0]
                    assert image.shape == label_image.shape
                    label_image = label_image.astype("uint32")

                    export_sam_model(
                        image=image,
                        label_image=label_image,
                        model_type=model_type,
                        name=self.name,
                        output_path=self.output_path,
                        checkpoint_path=export_checkpoint,
                    )

                pbar_signals.pbar_stop.emit()
                return self.output_path

            else:
                pbar_signals.pbar_stop.emit()
                return export_checkpoint

        path = run_training()
        print(f"Training has finished. The trained model is saved at {path}.")
        # worker = run_training()
        # worker.returned.connect(lambda path: print(f"Training has finished. The trained model is saved at {path}."))
        # worker.start()
        # return worker
