from util import compare_experiments_for_dataset


VIT_L_PARAMS = {
    "experiment_folder": "/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
    "standard_model": "vit_l",
    "finetuned_model": "vit_l_em_specialist",
    "intermediate_model": "vit_l_em_organelles_v2",
    "checkpoint1": None,
    "checkpoint3": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_l/mito_nuc_em_generalist_sam/best.pt"
}

SPECIALISTS = {
    "asem_er": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_l/asem_er_sam/best.pt",
    "cremi": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_l/cremi_sam/best.pt",
}


# HACK: the import issues don't matter, as long as the object satisfies the namespaces' requirement.
# class FilterObjectsLabelTrafo:
#     def __init__(self, min_size=0, gap_closing=0):
#         self.min_size = min_size
#         self.gap_closing = gap_closing

#     def __call__(self, labels):
#         if self.gap_closing > 0:
#             labels = binary_closing(labels, iterations=self.gap_closing)

#         distance_transform = PerObjectDistanceTransform(
#             distances=True,
#             boundary_distances=True,
#             directed_distances=False,
#             foreground=True,
#             instances=True,
#             min_size=self.min_size
#         )
#         labels = distance_transform(labels)
#         return labels


def compare_em_specialists():
    # for figure s8 (we use 'vit_l')
    all_datasets = ["asem_er", "cremi"]
    params = VIT_L_PARAMS

    for dataset in all_datasets:
        compare_experiments_for_dataset(dataset, checkpoint2=SPECIALISTS[dataset], **params)


def main():
    compare_em_specialists()


if __name__ == "__main__":
    main()
