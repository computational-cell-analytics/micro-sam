# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
microSAM is a state-of-the-art tool for interactive and automatic microscopy segmentation based on segment anything model (SAM).
The GUI is built as a napari plugin.
We are currently updating the tool to support SAM2 and we are refactoring and extending the GUI in this context. 
You will implement tasks related to this project.

## Common Commands

### Code Quality
```bash
# Format code with black
black micro_sam/

# Lint with ruff (auto-fixes enabled)
ruff check micro_sam/
```

### Tests
```bash
# Run all tests
# Note: this takes very long, only run the relevant tests for what you develop.
pytest
```

## Code Architecture

### Module Organization

**Core Segmentation (micro_sam/):**
- `util.py` - Model loading, device management, embeddings, preprocessing utilities
- `inference.py` - Batched/tiled inference for large images
- `prompt_based_segmentation.py` - Interactive segmentation with point/box/mask prompts
- `automatic_segmentation.py` - High-level API for automatic segmentation workflows
- `instance_segmentation.py` - Core automatic segmentation implementations (AMG, AIS, APG)
- `multi_dimensional_segmentation.py` - 3D volume and temporal tracking segmentation

**SAM v2 Support (micro_sam/v2/):**
- SAM v2 uses Hiera backbone (hvit_t, hvit_s, hvit_b, hvit_l) with temporal/video capabilities
- `v2/util.py` - SAM2-specific model loading and configuration
- `v2/prompt_based_segmentation.py` - Wrapper for SAM2 2D/3D predictions
- `v2/models/_video_predictor.py` - Video/tracking predictor
- Model type prefixes: SAM v1 = `vit_*`, SAM v2 = `hvit_*`

**Napari UI (micro_sam/sam_annotator/):**
- `_annotator.py` - Base annotator with napari layer/widget/keybinding setup
- `annotator_2d.py`, `annotator_3d.py`, `annotator_tracking.py` - Dimension-specific UIs
- `_state.py` - Singleton state manager (predictor, embeddings, AMG generators)
- `_widgets.py` - Qt widgets for embedding/segmentation/tracking controls
- `image_series_annotator.py` - Multi-image batch annotation workflow

**Training/Finetuning (micro_sam/training/):**
- `sam_trainer.py` - Base trainer extending torch_em.DefaultTrainer
- `joint_sam_trainer.py`, `simple_sam_trainer.py`, `semantic_sam_trainer.py` - Specialized trainers
- `training.py` - High-level training orchestration and CONFIGURATIONS registry
- `util.py` - Training data conversion and model loading

**Models (micro_sam/models/):**
- `build_sam.py` - Factory for SAM v1 models (vit_b, vit_l, vit_h, vit_t via MobileSAM)
- `peft_sam.py` - Parameter-efficient fine-tuning (LoRA, FacT, SSF, AdaptFormer)
- `sam_3d_wrapper.py` - 3D-compatible SAM wrapper
- `simple_sam_3d_wrapper.py` - Simplified 3D segmentation model

### Key Architectural Patterns

**Three Automatic Segmentation Modes:**

1. **AMG (Automatic Mask Generator)** - Default, grid-based prompting
   - Classes: `AutomaticMaskGenerator`, `TiledAutomaticMaskGenerator`
   - No decoder required
   - Factory: `get_instance_segmentation_generator(mode="amg")`

2. **AIS (Instance Segmentation with Decoder)** - UNETR decoder-based
   - Classes: `InstanceSegmentationWithDecoder`, `TiledInstanceSegmentationWithDecoder`
   - Requires trained decoder checkpoint
   - Factory: `get_instance_segmentation_generator(mode="ais")`

3. **APG (Automatic Prompt Generator)** - Decoder + iterative refinement
   - Classes: `AutomaticPromptGenerator`, `TiledAutomaticPromptGenerator`
   - Extends AIS with prompt refinement
   - Factory: `get_instance_segmentation_generator(mode="apg")`

All modes support tiling for large images via `inference.batched_tiled_inference()`.

**Precomputation and Caching:**
- `util.precompute_image_embeddings()` - Compute and cache embeddings
- `util.set_precomputed()` - Load precomputed embeddings
- `precompute_state.py` - CLI and batch precomputation
- Saves to zarr/h5 format for fast loading
- Embeddings stored as ImageEmbeddings dict with 'features', 'input_size', 'original_size'

**Prompt Generators (for training):**
- `PromptGeneratorBase` - Abstract interface
- `PointAndBoxPromptGenerator` - Samples prompts from ground-truth masks
- `IterativePromptGenerator` - Adapts prompts based on prediction errors
- Used by trainers for curriculum learning

**PEFT Surgery (Parameter-Efficient Fine-Tuning):**
- `PEFT_Sam` wrapper enables freezing most parameters
- Strategies: LoRA, FacT, SSF, AdaptFormer, ClassicalSurgery
- Configured via `models.peft_sam.PEFT_Sam(sam_model, rank=4, peft_module="lora")`

### Data Flow

**Interactive Annotation:**
```
User input (napari) → AnnotatorState → Predictor.predict() →
Update napari layers → Display result
```

**Automatic Segmentation:**
```
Image → util.precompute_image_embeddings() →
AMG/AIS/APG.initialize() → generator.generate() →
Instance masks
```

**Training:**
```
DataLoader → ConvertToSamInputs → SamTrainer →
Iterative prompting → Loss (Dice + IoU MSE) →
Save checkpoint with decoder_state
```

### Important Implementation Notes

**Model Registry and Loading:**
- SAM v1 models downloaded from Facebook via `util.get_sam_model()`
- SAM v2 models via `v2.util.get_sam2_model()`
- Finetuned models available: `vit_b_lm` (light microscopy), `vit_b_em_organelles`, etc.
- Model type determines architecture automatically

**State Management:**
- `AnnotatorState` is a singleton (metaclass-based)
- Shared across all annotator instances
- Contains predictor, embeddings, AMG generator, decoder

**Tiling Strategy:**
- Enabled by `is_tiled=True` in factory functions
- Applies halos for overlap handling
- Merges results avoiding duplicates
- Critical for large images (>2048px)

**SAM v1 vs v2 Routing:**
- Model type prefix determines version: `vit_*` → SAM v1, `hvit_*` → SAM v2
- `_state._get_sam_model()` handles version selection
- SAM v2 adds video/tracking capabilities via video predictor

**Training Checkpoint Format:**
```python
{
    'model': model_state_dict,
    'decoder_state': decoder_weights,  # Optional, for AIS/APG modes
    'config': model_config,
    'epoch': int,
    'optimizer': optimizer_state
}
```

**Decoder Integration:**
- UNETR decoder predicts: center distances, boundaries, foreground probability
- Loaded via `get_decoder()` from checkpoint's `decoder_state`
- Used by AIS and APG modes for prompt generation

### Testing Guidelines

- Use google style docstrings for new code
- Write unit tests for new functionality
- GUI tests should use `make_napari_viewer_proxy` fixture
- Mark slow tests: `@pytest.mark.slow`
- Mark GUI tests: `@pytest.mark.gui`
- Coverage reports generated automatically with pytest-cov

### Environment Variables

- `PYTORCH_ENABLE_MPS_FALLBACK` - Enable Apple Silicon MPS fallback
