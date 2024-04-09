import os
import argparse
import warnings

import pandas as pd

from elf.evaluation import mean_segmentation_accuracy

from micro_sam import instance_segmentation
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search


def _3d_automatic_instance_segmentation_with_decoder(
    test_raw, test_labels, model_type, checkpoint_path,
    result_dir, embedding_dir, auto_3d_seg_kwargs, species=None
):
    # let's run ais on the test volume
    res_dir = os.path.join(result_dir, "" if species is None else species, "auto")
    res_path = os.path.join(res_dir, "ais.csv")
    if os.path.exists(res_dir):
        # we assume that the inference has completed
        print("The results are already stored at:", res_path)
        return
    else:
        os.makedirs(res_dir, exist_ok=True)

    predictor, decoder = instance_segmentation.get_predictor_and_decoder(model_type, checkpoint_path)
    segmentor = instance_segmentation.InstanceSegmentationWithDecoder(predictor, decoder)

    _embedding_path = os.path.join(embedding_dir, "" if species is None else species, "test")

    instances = automatic_3d_segmentation(
        volume=test_raw,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=_embedding_path,
        **auto_3d_seg_kwargs
    )

    msa, sa = mean_segmentation_accuracy(instances, test_labels, return_accuracies=True)
    print("mSA for 3d volume is:", msa)
    print("SA50 for 3d volume is:", sa[0])
    save_auto_res = {
        "mSA": msa, "SA50": sa[0]
    }
    save_auto_resdf = pd.DataFrame.from_dict([save_auto_res])
    save_auto_resdf.to_csv(res_path)
    print("The results have been computed and stored at:", res_path)


def _3d_interactive_instance_segmentation(
    val_raw, val_labels, test_raw, test_labels, model_type, checkpoint_path,
    result_dir, embedding_dir, species=None, min_size=50
):
    # let's do grid-search on the val set
    _val_embedding_path = os.path.join(embedding_dir, "" if species is None else species, "val")
    _val_res_dir = os.path.join(result_dir, "" if species is None else species, "interactive", "val")
    best_params_path = run_multi_dimensional_segmentation_grid_search(
        volume=val_raw,
        ground_truth=val_labels,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        embedding_path=_val_embedding_path,
        result_dir=_val_res_dir,
        interactive_seg_mode="box",
        verbose=False,
        min_size=min_size,
    )

    best_params = {}
    resdf = pd.read_csv(best_params_path)
    for k, v in resdf.loc[0].items():
        if k.startswith("Unnamed") or k == "mSA":
            continue
        best_params[k] = [v]

    # now let's use the best parameters on the test set
    _test_embedding_path = os.path.join(embedding_dir, "" if species is None else species, "test")
    _test_res_dir = os.path.join(result_dir, "" if species is None else species, "interactive", "test")
    run_multi_dimensional_segmentation_grid_search(
        volume=test_raw,
        ground_truth=test_labels,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        embedding_path=_test_embedding_path,
        result_dir=_test_res_dir,
        interactive_seg_mode="box",
        verbose=False,
        grid_search_values=best_params,
        min_size=min_size,
    )


def _get_default_args(input_path):
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default=input_path, help="Path to volume."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path."
    )
    parser.add_argument(
        "-e", "--experiment_folder", type=str, default="./experiment_folder",
        help="Path where the embeddings and results will be stored."
    )
    parser.add_argument(
        "--ais", action="store_true", help="Whether to perforn 3d ais."
    )
    parser.add_argument(
        "--int", action="store_true", help="Whether to perform 3d interactive instance segmentation."
    )
    parser.add_argument(
        "--species", type=str, default=None, help="Relevant for MitoEM."
    )
    args = parser.parse_args()
    return args
