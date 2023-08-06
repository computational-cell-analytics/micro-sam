import argparse
from micro_sam.evaluation.livecell import run_livecell_amg
from util import DATA_ROOT, get_checkpoint, get_experiment_folder, check_model


def run_job(model_name, use_mws):
    checkpoint, model_type = get_checkpoint(model_name)
    experiment_folder = get_experiment_folder(model_name)
    input_folder = DATA_ROOT

    run_livecell_amg(
        checkpoint, model_type, input_folder, experiment_folder,
        n_val_per_cell_type=25, use_mws=use_mws,
    )


# TODO
def check_amg(model_name, use_mws):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("--mws", action="store_true")
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    model_name = args.name
    use_mws = args.mws
    check_model(model_name)

    if args.check:
        check_amg(model_name, use_mws)
    else:
        run_job(model_name, use_mws)


if __name__ == "__main__":
    main()
