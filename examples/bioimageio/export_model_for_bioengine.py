"""Export a model for the BioEngine. The export supports the SAM1 models of micro-sam v1 only."""
from micro_sam.v1.bioimageio.bioengine_export import export_bioengine_model

export_bioengine_model("vit_b", "test-export", opset=12)
