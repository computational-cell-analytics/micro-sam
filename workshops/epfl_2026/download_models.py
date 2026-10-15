from micro_sam.util import get_model_cache_directory
from micro_sam.v1.util import get_sam_model

get_sam_model(model_type="vit_b_lm")
get_sam_model(model_type="vit_b_em_organelles")
print(f"The models for the workshop have been downloaded to {get_model_cache_directory()}")
