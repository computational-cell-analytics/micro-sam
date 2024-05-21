import os
import warnings

import torch
import torch_em
from napari.qt.threading import thread_worker
from qtpy import QtWidgets
from torch.utils.data import random_split

import micro_sam.util as util
import micro_sam.sam_annotator._widgets as widgets
from ._tooltips import get_tooltip
from micro_sam.training import default_sam_dataset, train_sam_for_configuration, CONFIGURATIONS


def _find_best_configuration():
    if torch.cuda.is_available():

        # Check how much memory we have and select the best matching GPU
        # for the available VRAM size.
        _, vram = torch.cuda.mem_get_info()
        vram = vram / 1e9  # in GB

        # Maybe we can get more configurations in the future.
        if vram > 80:  # More than 80 GB: use the A100 configurations.
            return "A100"
        elif vram > 30:  # More than 30 GB: use the V100 configurations.
            return "V100"
        elif vram > 14:  # More than 14 GB: use the RTX5000 configurations.
            return "rtx5000"
        else:  # Otherwise: not enough memory to train on the GPU, use CPU instead.
            return "CPU"
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
            "Path to images", self.raw_path, "both", placeholder="/path/to/images",
            tooltip=get_tooltip("training", "raw_path")
        )
        self.layout().addLayout(layout)

        self.raw_key = None
        _, layout = self._add_string_param(
            "Image data key", self.raw_key, placeholder="e.g. \"*.tif\"",
            tooltip=get_tooltip("training", "raw_key")
        )
        self.layout().addLayout(layout)

        self.label_path = None
        _, layout = self._add_path_param(
            "Path to labels", self.label_path, "both", placeholder="/path/to/labels",
            tooltip=get_tooltip("training", "label_path")
        )
        self.layout().addLayout(layout)

        self.label_key = None
        _, layout = self._add_string_param(
            "Label data key", self.label_key, placeholder="e.g. \"*.tif\"",
            tooltip=get_tooltip("training", "label_key")
        )
        self.layout().addLayout(layout)

        self.configuration = _find_best_configuration()
        self.setting_dropdown, layout = self._add_choice_param(
            "Configuration", self.configuration, list(CONFIGURATIONS.keys()),
            tooltip=get_tooltip("training", "configuration")
        )
        self.layout().addLayout(layout)

        self.with_segmentation_decoder = True
        self.layout().addWidget(self._add_boolean_param(
            "With segmentation decoder", self.with_segmentation_decoder,
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
            "Device", self.device, device_options, tooltip=get_tooltip("training", "device")
        )
        setting_values.layout().addLayout(layout)

        self.patch_x, self.patch_y = 512, 512
        self.patch_x_param, self.patch_y_param, layout = self._add_shape_param(
            ("Patch size x", "Patch size y"), (self.patch_x, self.patch_y), min_val=0, max_val=2048,
            tooltip=get_tooltip("training", "patch")
        )
        setting_values.layout().addLayout(layout)

        # Paths for validation data.
        self.raw_path_val = None
        _, layout = self._add_path_param(
            "Path to validation images", self.raw_path_val, "both", placeholder="/path/to/images",
            tooltip=get_tooltip("training", "raw_path_val")
        )
        setting_values.layout().addLayout(layout)

        self.label_path_val = None
        _, layout = self._add_path_param(
            "Path to validation labels", self.label_path_val, "both", placeholder="/path/to/images",
            tooltip=get_tooltip("training", "label_path_val")
        )
        setting_values.layout().addLayout(layout)

        # Name of the model to be trained and options to over-ride the initial model
        # on top of which the finetuning is run.
        self.name = "sam_model"
        self.name_param, layout = self._add_string_param(
            "Model name", self.name, tooltip=get_tooltip("training", "name")
        )
        setting_values.layout().addLayout(layout)

        self.initial_model = None
        self.initial_model_param, layout = self._add_string_param(
            "Initial model", self.initial_model, tooltip=get_tooltip("training", "initial_model")
        )
        setting_values.layout().addLayout(layout)

        self.checkpoint = None
        self.checkpoint_param, layout = self._add_string_param(
            "Checkpoint", self.name, tooltip=get_tooltip("training", "checkpoint")
        )
        setting_values.layout().addLayout(layout)

        self.output_path = None
        self.output_path_param, layout = self._add_string_param(
            "Output Path", self.output_path, tooltip=get_tooltip("training", "output_path")
        )
        setting_values.layout().addLayout(layout)

        self.n_epochs = 100
        self.n_epochs_param, layout = self._add_int_param(
            "Number of epochs", self.n_epochs, tooltip=get_tooltip("training", "n_epochs"),
            min_val=1, max_val=1000,
        )
        setting_values.layout().addLayout(layout)

        settings = widgets._make_collapsible(setting_values, title="Advanced Settings")
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
            # Use 10% of the dataset - at least one image - for validation.
            n_val = min(1, int(0.1 * len(dataset)))
            train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - n_val, n_val])
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

        @thread_worker()
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

                    # Select the last channel of the label image if we have a channel axis.
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

        worker = run_training()
        worker.returned.connect(lambda path: print(f"Training has finished. The trained model is saved at {path}."))
        worker.start()
        return worker
