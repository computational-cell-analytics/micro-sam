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
@click.option("--tile_shape", type=int, nargs=2, default=None, help="The tile shape for tiled prediction.")
@click.option("--halo", type=int, nargs=2, default=None, help="The halo for tiled prediction.")
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
    from .v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

    model_type = model_type or DEFAULT_MODEL
    tile_shape = tile_shape or None
    halo = halo or None

    def _convert_argval(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    extra = ctx.args
    generate_kwargs = {extra[i].lstrip("-"): _convert_argval(extra[i + 1]) for i in range(0, len(extra), 2)}

    predictor, segmenter = get_predictor_and_segmenter(
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
            predictor=predictor,
            segmenter=segmenter,
            input_path=path,
            output_path=output_fpath,
            embedding_path=embedding_fpath,
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


def _passthrough(module_path):
    """Build a click command that forwards all raw args to a legacy argparse ``main(argv)``."""

    @click.command(
        context_settings=dict(ignore_unknown_options=True, help_option_names=[]), add_help_option=False,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args):
        import importlib
        importlib.import_module(module_path).main(list(args))

    return command


v1.add_command(_passthrough("micro_sam.v1.training.training"), "train")
v1.add_command(_passthrough("micro_sam.v1.automatic_segmentation"), "automatic_segmentation")
v1.add_command(_passthrough("micro_sam.v1.evaluation.evaluation"), "evaluate")
v1.add_command(_passthrough("micro_sam.v1.evaluation.benchmark_datasets"), "benchmark_sam")


if __name__ == "__main__":
    cli()
