from micro_sam.util import get_sam_model, microsam_cachedir

get_sam_model(model_type="vit_b_lm")
get_sam_model(model_type="vit_b_em_organelles")
print(f"The models for the workshop have been downloaded to {microsam_cachedir()}")
