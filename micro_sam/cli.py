"""Command line interface for micro_sam.

A single ``micro_sam`` entry point exposes the current (SAM2 / v2) functionality as subcommands,
with the legacy SAM1 (v1) tooling grouped under ``micro_sam v1``. Command callbacks import their
heavy dependencies (torch, napari, ...) lazily, so ``micro_sam --help`` stays fast.
"""

import click


@click.group()
def cli():
    """micro_sam: Segment Anything for Microscopy."""


@cli.group("v1")
def v1():
    """Legacy SAM1 (micro_sam v1) tooling."""


def _interactive_options(f):
    """Common options shared by the interactive annotator commands."""
    options = [
        click.option(
            "-i", "--input", "input_", required=True,
            help="The filepath to the image data. Supports all data types readable by imageio (e.g. tif, png, ...) "
            "or elf.io.open_file (e.g. hdf5, zarr, mrc). For the latter also pass '--key'."
        ),
        click.option(
            "-k", "--key", default=None,
            help="The key for opening data with elf.io.open_file, e.g. the internal path for hdf5 / zarr, "
            "a wildcard like '*.png' for an image stack, or 'data' for mrc."
        ),
        click.option(
            "-e", "--embedding_path", default=None,
            help="Filepath for saving / loading the pre-computed image embeddings. Recommended to reuse embeddings "
            "across sessions; otherwise they are recomputed every time."
        ),
        click.option(
            "-m", "--model_type", default=None,
            help="The Segment Anything model to use. By default the finetuned SAM2 model 'hvit_t_cells' is used."
        ),
        click.option("-c", "--checkpoint", "checkpoint_path", default=None, help="Checkpoint to load the model from."),
        click.option(
            "--decoder_path", default=None,
            help="Optional decoder-only weights to enable decoder-based instance segmentation."
        ),
        click.option(
            "-d", "--device", default=None,
            help="The device for the predictor: 'cuda', 'cpu' or 'mps'. By default the best available is used."
        ),
        click.option("--tile_shape", type=int, nargs=2, default=None, help="The tile shape for tiled prediction."),
        click.option("--halo", type=int, nargs=2, default=None, help="The halo for tiled prediction."),
    ]
    for option in reversed(options):
        f = option(f)
    return f


@cli.command("segmentation_annotator")
@click.option(
    "-s", "--segmentation_result", default=None,
    help="Optional filepath to a precomputed segmentation to initialize the 'committed_objects' layer. "
    "Supports the same file formats as '--input'."
)
@click.option(
    "-sk", "--segmentation_key", default=None, help="The key for opening the segmentation data. Same rules as '--key'."
)
@click.option(
    "--ndim", type=int, default=None,
    help="The number of spatial dimensions (2 or 3). If not given, auto-detected from the image shape."
)
@click.option(
    "--precompute_amg_state", is_flag=True, default=False,
    help="Whether to precompute the state for automatic instance segmentation (longer start-up, faster first run)."
)
@click.option(
    "--prefer_decoder", is_flag=True, default=True, flag_value=False,
    help="Whether to use decoder based instance segmentation if the model has an additional decoder for that purpose."
)
@_interactive_options
def segmentation_annotator(
    input_, key, embedding_path, model_type, checkpoint_path, decoder_path, device, tile_shape, halo,
    segmentation_result, segmentation_key, ndim, precompute_amg_state, prefer_decoder,
):
    """Start the segmentation annotator for 2D or 3D image data."""
    from .util import load_image_data
    from .sam_annotator.annotator import annotator
    from .v2.util import DEFAULT_MODEL

    image = load_image_data(input_, key=key)
    segmentation = None if segmentation_result is None else load_image_data(segmentation_result, key=segmentation_key)

    annotator(
        image,
        ndim=ndim,
        embedding_path=embedding_path,
        segmentation_result=segmentation,
        model_type=model_type or DEFAULT_MODEL,
        tile_shape=tile_shape or None,
        halo=halo or None,
        precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path,
        decoder_path=decoder_path,
        device=device,
        prefer_decoder=prefer_decoder,
    )


@cli.command("tracking_annotator")
@_interactive_options
def tracking_annotator(
    input_, key, embedding_path, model_type, checkpoint_path, decoder_path, device, tile_shape, halo,
):
    """Start the tracking annotator for a timeseries."""
    from .util import load_image_data
    from .sam_annotator.annotator_tracking import annotator_tracking
    from .v2.util import DEFAULT_MODEL

    image = load_image_data(input_, key=key)
    annotator_tracking(
        image,
        embedding_path=embedding_path,
        model_type=model_type or DEFAULT_MODEL,
        tile_shape=tile_shape or None,
        halo=halo or None,
        checkpoint_path=checkpoint_path,
        decoder_path=decoder_path,
        device=device,
    )


@cli.command("batch_annotator")
@click.option("-i", "--input_folder", required=True, help="The folder containing the image data (tif, jpg, png, ...).")
@click.option("-o", "--output_folder", required=True, help="The folder where the segmentation results will be stored.")
@click.option(
    "--ndim", type=int, default=None,
    help="The number of spatial dimensions (2 or 3). If not given, auto-detected from the image shape."
)
@click.option(
    "-p", "--pattern", default="*",
    help="Glob pattern to select images from the input folder, e.g. '*.tif'. By default all files are loaded."
)
@click.option(
    "--initial_segmentation_folder", default=None, help="A folder with initial segmentation results to load."
)
@click.option(
    "--initial_segmentation_pattern", default="*", help="The glob pattern for '--initial_segmentation_folder'."
)
@click.option(
    "-e", "--embedding_path", default=None,
    help="Filepath for saving / loading the pre-computed image embeddings. Recommended to avoid recomputation."
)
@click.option(
    "-m", "--model_type", default=None,
    help="The Segment Anything model to use. By default the finetuned SAM2 model 'hvit_t_cells' is used."
)
@click.option("-c", "--checkpoint", "checkpoint_path", default=None, help="Checkpoint to load the model from.")
@click.option(
    "-d", "--device", default=None,
    help="The device for the predictor: 'cuda', 'cpu' or 'mps'. By default the best available is used."
)
@click.option("--tile_shape", type=int, nargs=2, default=None, help="The tile shape for tiled prediction.")
@click.option("--halo", type=int, nargs=2, default=None, help="The halo for tiled prediction.")
@click.option("--precompute_amg_state", is_flag=True, default=False, help="Whether to precompute the AMG state.")
@click.option(
    "--prefer_decoder", is_flag=True, default=True, flag_value=False,
    help="Whether to use decoder based instance segmentation if the model has an additional decoder for that purpose."
)
@click.option(
    "--skip_segmented", is_flag=True, default=True, flag_value=False,
    help="Whether to skip images that were already segmented."
)
def batch_annotator(
    input_folder, output_folder, ndim, pattern, initial_segmentation_folder, initial_segmentation_pattern,
    embedding_path, model_type, checkpoint_path, device, tile_shape, halo, precompute_amg_state, prefer_decoder,
    skip_segmented,
):
    """Annotate a batch of images from a folder."""
    from .sam_annotator.batch_annotator import image_folder_annotator
    from .v2.util import DEFAULT_MODEL

    image_folder_annotator(
        input_folder, output_folder, pattern=pattern, ndim=ndim,
        initial_segmentation_folder=initial_segmentation_folder,
        initial_segmentation_pattern=initial_segmentation_pattern,
        embedding_path=embedding_path, model_type=model_type or DEFAULT_MODEL,
        tile_shape=tile_shape or None, halo=halo or None, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device,
        prefer_decoder=prefer_decoder, skip_segmented=skip_segmented,
    )


@cli.command("precompute_embeddings")
@click.option("-i", "--input_path", required=True, help="The filepath to the image data (also container files).")
@click.option("-e", "--embedding_path", required=True, help="The path where the embeddings will be saved.")
@click.option("--pattern", default=None, help="Glob pattern to select files in a folder, e.g. '*'.")
@click.option("-k", "--key", default=None, help="The key for opening data with elf.io.open_file.")
@click.option(
    "-m", "--model_type", default=None,
    help="The SAM2 model to use. By default the base backbone 'hvit_t' is used."
)
@click.option("-c", "--checkpoint", "checkpoint_path", default=None, help="Checkpoint to load the SAM2 model from.")
@click.option(
    "-n", "--ndim", type=int, default=None,
    help="The number of spatial dimensions. Specify this if your data has a channel dimension."
)
def precompute_embeddings(input_path, embedding_path, pattern, key, model_type, checkpoint_path, ndim):
    """Precompute and cache the SAM2 image embeddings for image data."""
    from .precompute_state import precompute_state
    from .v2.util import _DEFAULT_MODEL

    precompute_state(
        input_path, embedding_path,
        model_type=model_type or _DEFAULT_MODEL, checkpoint_path=checkpoint_path,
        pattern=pattern, key=key, ndim=ndim,
    )


@cli.command(
    "automatic_segmentation", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "-i", "--input_path", required=True, multiple=True, help="The filepath(s) to the image data (also container files)."
)
@click.option(
    "-o", "--output_path", required=True,
    help="The filepath to store the results. For multiple inputs this should be a folder; "
    "for a single image a tif file."
)
@click.option("-e", "--embedding_path", default=None, help="Optional path where the embeddings will be cached.")
@click.option("--pattern", default=None, help="Glob pattern to select files in a folder, e.g. '*'.")
@click.option("-k", "--key", default=None, help="The key for opening data with elf.io.open_file.")
@click.option(
    "-m", "--model_type", default=None,
    help="The SAM2 model to use. Needs a registered decoder or a '--checkpoint'. Default: 'hvit_t_cells'."
)
@click.option("-c", "--checkpoint", "checkpoint_path", default=None, help="Decoder checkpoint to load the model from.")
@click.option(
    "--tile_shape", default=None,
    help="The tile shape for tiled prediction, comma-separated, e.g. '384,384' (2D) or '4,384,384' (3D)."
)
@click.option(
    "--halo", default=None,
    help="The halo for tiled prediction, comma-separated, e.g. '64,64' (2D) or '2,64,64' (3D)."
)
@click.option(
    "-n", "--ndim", type=int, default=None,
    help="The number of spatial dimensions. Specify this if your data has a channel dimension."
)
@click.option(
    "--mode", default="sparse", type=click.Choice(["sparse", "dense"]),
    help="The segmentation mode: 'sparse' (flow, LM data) or 'dense' (multicut, EM data)."
)
@click.option(
    "-d", "--device", default=None,
    help="The device for the predictor: 'cuda', 'cpu' or 'mps'. By default the best available is used."
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Whether to allow verbosity of outputs.")
@click.pass_context
def automatic_segmentation(
    ctx, input_path, output_path, embedding_path, pattern, key, model_type, checkpoint_path,
    tile_shape, halo, ndim, mode, device, verbose,
):
    """Run SAM2 automatic instance segmentation for 2D or 3D data.

    Additional postprocessing parameters (e.g. '--foreground_threshold' for sparse or '--beta' for
    dense) can be passed through to the segmentation and are forwarded to the segmenter.
    """
    import os
    from pathlib import Path

    from tqdm import tqdm

    from .v2.util import DEFAULT_MODEL
    from .v1.automatic_segmentation import _get_inputs_from_paths
    from .v2.automatic_segmentation import get_segmenter, automatic_instance_segmentation

    model_type = model_type or DEFAULT_MODEL

    def _parse_shape(value):
        if value is None:
            return None
        return tuple(int(x) for x in value.replace(" ", "").split(","))

    tile_shape = _parse_shape(tile_shape)
    halo = _parse_shape(halo)

    def _convert_argval(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def _parse_extra(tokens):
        # Pass-through postprocessing options, supporting both '--key value' and '--key=value'.
        kwargs, i = {}, 0
        while i < len(tokens):
            token = tokens[i]
            if not token.startswith("-"):
                raise click.UsageError(f"Expected an option starting with '--', got '{token}'.")
            if "=" in token:
                name, value = token.lstrip("-").split("=", 1)
                i += 1
            elif i + 1 < len(tokens):
                name, value, i = token.lstrip("-"), tokens[i + 1], i + 2
            else:
                raise click.UsageError(f"Missing value for option '{token}'.")
            kwargs[name] = _convert_argval(value)
        return kwargs

    generate_kwargs = _parse_extra(ctx.args)

    segmenter = get_segmenter(
        model_type=model_type, checkpoint=checkpoint_path, device=device, is_tiled=tile_shape is not None,
    )

    input_paths = _get_inputs_from_paths(list(input_path), pattern)
    assert len(input_paths) > 0, "'micro-sam' could not extract any image data internally."
    has_one_input = len(input_paths) == 1

    for path in tqdm(input_paths, desc="Run automatic segmentation"):
        if has_one_input:
            embedding_fpath = embedding_path
            output_fpath = f"{os.path.splitext(output_path)[0]}.tif"
            output_dir = os.path.dirname(output_fpath)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            input_name = str(Path(path).stem)
            if embedding_path is None:
                embedding_fpath = None
            else:
                embedding_folder = os.path.splitext(embedding_path)[0]
                os.makedirs(embedding_folder, exist_ok=True)
                embedding_fpath = os.path.join(embedding_folder, f"{input_name}.zarr")
            output_folder = os.path.splitext(output_path)[0]
            os.makedirs(output_folder, exist_ok=True)
            output_fpath = os.path.join(output_folder, f"{input_name}.tif")

        automatic_instance_segmentation(
            segmenter=segmenter,
            input_path=path,
            output_path=output_fpath,
            embedding_path=embedding_fpath,
            model_type=model_type,
            checkpoint=checkpoint_path,
            key=key,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
            mode=mode,
            device=device,
            verbose=verbose,
            **generate_kwargs,
        )


@cli.command("train")
@click.option("-c", "--config", required=True, help="The filepath to the SAM2 training config file.")
@click.option("--use_cluster", type=int, default=None, help="Whether to launch on a cluster: 0 local, 1 cluster.")
@click.option("--partition", default=None, help="SLURM partition.")
@click.option("--account", default=None, help="SLURM account.")
@click.option("--qos", default=None, help="SLURM qos.")
@click.option("--num_gpus", type=int, default=None, help="Number of GPUs per node.")
@click.option("--num_nodes", type=int, default=None, help="Number of nodes.")
def train(config, use_cluster, partition, account, qos, num_gpus, num_nodes):
    """Train a SAM2 model."""
    from .v2.train import train_sam2, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    train_sam2(
        config=config,
        use_cluster=bool(use_cluster) if use_cluster is not None else None,
        partition=partition,
        account=account,
        qos=qos,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
    )


@cli.command("info")
@click.option(
    "--download", multiple=True,
    help="Download pretrained SAM models, e.g. '--download models' or '--download models vit_b_lm'."
)
def info(download):
    """Display micro_sam information (version, cache, models, system)."""
    from .util import micro_sam_info
    micro_sam_info(download=list(download) if download else None)


def _delegate_argv(ctx):
    """Rebuild an argv list from parsed click options, to forward to a legacy argparse ``main(argv)``.

    Value options are emitted only when given (so argparse supplies its own defaults), 'store_true'
    flags only when set, and multi-value options as a single flag followed by the values (matching
    argparse ``nargs``). Unknown pass-through tokens (e.g. the automatic-segmentation postprocessing
    parameters) are appended verbatim.
    """
    argv = []
    for param in ctx.command.params:
        if not isinstance(param, click.Option):
            continue
        value = ctx.params.get(param.name)
        long_opt = next((opt for opt in param.opts if opt.startswith("--")), param.opts[0])
        if param.is_flag:
            if value:
                argv.append(long_opt)
        elif value is None:
            continue
        elif param.multiple:
            values = list(value)
            if values:
                argv.append(long_opt)
                argv.extend(str(v) for v in values)
        else:
            argv.extend([long_opt, str(value)])
    argv.extend(ctx.args)
    return argv


def _run_legacy(module, prog, ctx):
    """Forward the parsed options to a legacy v1 argparse ``main(argv)``, with a proper program name."""
    import sys
    argv = _delegate_argv(ctx)
    original_argv0 = sys.argv[0]
    sys.argv[0] = prog  # so argparse usage / error messages read 'micro_sam v1 <cmd>'.
    try:
        module.main(argv)
    finally:
        sys.argv[0] = original_argv0


@v1.command("train")
@click.option("--images", multiple=True, required=True, help="Filepath(s) to images or the image directory.")
@click.option("--labels", multiple=True, required=True, help="Filepath(s) to labels or the label directory.")
@click.option("--image_key", default=None, help="The key for accessing image data (pattern or elf.io.open_file key).")
@click.option("--label_key", default=None, help="The key for accessing label data (pattern or elf.io.open_file key).")
@click.option("--val_images", multiple=True, help="Filepath(s) to validation images or the directory.")
@click.option("--val_labels", multiple=True, help="Filepath(s) to validation labels or the directory.")
@click.option("--val_image_key", default=None, help="The key for accessing validation image data.")
@click.option("--val_label_key", default=None, help="The key for accessing validation label data.")
@click.option(
    "--configuration", default=None,
    help="The finetuning configuration. By default the best for the available hardware is used."
)
@click.option(
    "--segmentation_decoder", default=None,
    help="Whether to also train a segmentation decoder: 'instances', 'instances_only' or 'None'. Default 'instances'."
)
@click.option(
    "-d", "--device", default=None,
    help="The device for finetuning: 'cuda', 'cpu' or 'mps'. By default the best available is used."
)
@click.option("--patch_shape", multiple=True, type=int, help="The patch shape for training. Default '512 512'.")
@click.option("-m", "--model_type", default=None, help="The Segment Anything model to finetune.")
@click.option("--checkpoint_path", default=None, help="Checkpoint to load the SAM model from for finetuning.")
@click.option("-s", "--save_root", default=None, help="Directory to store trained models and logs. By default the cwd.")
@click.option("--trained_model_name", default=None, help="The trained model sub-folder name. Default 'sam_model'.")
@click.option("--output_path", default=None, help="The directory or filepath to export the trained model to.")
@click.option("--n_epochs", type=int, default=None, help="The number of epochs to train. Default 100.")
@click.option("--num_workers", type=int, default=None, help="The number of dataloader workers. Default 1.")
@click.option("--batch_size", type=int, default=None, help="The training batch size. Default 1.")
@click.option(
    "--preprocess", type=click.Choice(["normalize_minmax", "normalize_percentile"]), default=None,
    help="Optional input normalization. By default no normalization is applied."
)
@click.pass_context
def v1_train(ctx, **kwargs):
    """Finetune a SAM1 model on custom data."""
    from .v1.training import training
    _run_legacy(training, "micro_sam v1 train", ctx)


@v1.command("automatic_segmentation", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "-i", "--input_path", multiple=True, required=True, help="The filepath(s) to the image data (also container files)."
)
@click.option(
    "-o", "--output_path", required=True,
    help="The filepath to store the results. For multiple inputs a folder; for a single image a tif file."
)
@click.option("-e", "--embedding_path", default=None, help="Optional path where the embeddings will be cached.")
@click.option("--pattern", default=None, help="Glob pattern to select files in a folder, e.g. '*'.")
@click.option("-k", "--key", default=None, help="The key for opening data with elf.io.open_file.")
@click.option("-m", "--model_type", default=None, help="The Segment Anything model to use.")
@click.option("-c", "--checkpoint", default=None, help="Checkpoint to load the SAM model from.")
@click.option("--tile_shape", multiple=True, type=int, help="The tile shape for tiled prediction, e.g. '384 384'.")
@click.option("--halo", multiple=True, type=int, help="The halo for tiled prediction, e.g. '64 64'.")
@click.option("-n", "--ndim", type=int, default=None, help="The number of spatial dimensions in the data.")
@click.option("--mode", default=None, help="The automatic segmentation mode: 'auto', 'amg', 'ais' or 'apg'.")
@click.option("--annotate", is_flag=True, default=False, help="Whether to continue annotation after segmentation.")
@click.option(
    "-d", "--device", default=None,
    help="The device for the predictor: 'cuda', 'cpu' or 'mps'. By default the best available is used."
)
@click.option("--batch_size", type=int, default=None, help="The batch size for computing embeddings over tiles / z.")
@click.option("--tracking", is_flag=True, default=False, help="Run automatic tracking instead of segmentation.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Whether to allow verbosity of outputs.")
@click.pass_context
def v1_automatic_segmentation(ctx, **kwargs):
    """Run SAM1 automatic segmentation or tracking for 2D, 3D or timeseries data.

    Additional postprocessing parameters (e.g. '--pred_iou_thresh' for AMG or '--center_distance_threshold'
    for AIS / APG) can be passed through and are forwarded to the chosen segmentation mode.
    """
    from .v1 import automatic_segmentation
    _run_legacy(automatic_segmentation, "micro_sam v1 automatic_segmentation", ctx)


@v1.command("evaluate")
@click.option("--labels", multiple=True, required=True, help="Filepath(s) to ground-truth labels or the directory.")
@click.option("--predictions", multiple=True, required=True, help="Filepath(s) to predicted labels or the directory.")
@click.option("--label_key", default=None, help="The key for accessing label data (pattern or elf.io.open_file key).")
@click.option(
    "--prediction_key", default=None, help="The key for accessing prediction data (pattern or elf.io.open_file key)."
)
@click.option("-o", "--output_path", default=None, help="The filepath to store the evaluation results (a csv file).")
@click.option(
    "--threshold", multiple=True, type=float,
    help="Overlap threshold(s) for the segmentation accuracy. By default np.arange(0.5, 1., 0.05) is used."
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Whether to allow verbosity of evaluation.")
@click.pass_context
def v1_evaluate(ctx, **kwargs):
    """Evaluate instance segmentations against ground-truth with SAM1 tooling."""
    from .v1.evaluation import evaluation
    _run_legacy(evaluation, "micro_sam v1 evaluate", ctx)


@v1.command("benchmark_sam")
@click.option(
    "-i", "--input_folder", required=True, help="Directory where the microscopy datasets are and/or will be stored."
)
@click.option("-m", "--model_type", default=None, help="The segment anything model that will be used.")
@click.option("-c", "--checkpoint_path", default=None, help="Checkpoint to load the SAM model from.")
@click.option(
    "-d", "--dataset_choice", multiple=True,
    help="The dataset(s) to evaluate on. Multiple can be given. By default all datasets are evaluated."
)
@click.option("-o", "--output_folder", required=True, help="The path where the results will be stored as csv files.")
@click.option("--amg", is_flag=True, default=False, help="Whether to run automatic segmentation in AMG mode.")
@click.option(
    "--retain", multiple=True,
    help="Parts of the benchmark to retain: one or more of 'data', 'crops', 'automatic', 'interactive'."
)
@click.option(
    "--evaluate", type=click.Choice(["all", "automatic", "interactive"]), default=None,
    help="The methods to benchmark: 'all', 'automatic' or 'interactive'. Default 'all'."
)
@click.pass_context
def v1_benchmark_sam(ctx, **kwargs):
    """Benchmark Segment Anything models on microscopy datasets."""
    from .v1.evaluation import benchmark_datasets
    _run_legacy(benchmark_datasets, "micro_sam v1 benchmark_sam", ctx)


if __name__ == "__main__":
    cli()
