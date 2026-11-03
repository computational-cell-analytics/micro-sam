"""Default parameters per model.

The SAM1 tables keep the earlier settings of the `vit_*` models for the SAM1 backend.
The SAM2 tables hold the defaults of the `hvit_*` models for automatic instance segmentation (AIS)
and for automatic prompt generation (APG).
"""

# The settings for the instance segmentation widget with ais.
AIS_SETTINGS = {
    "vit_t_lm": {
        "center_distance_thresh": 0.5,
        "boundary_distance_thresh": 0.5,
        "distance_smoothing": 2.0,
        "min_size": 100,
    },
    "vit_b_lm": {
        "center_distance_thresh": 0.4,
        "boundary_distance_thresh": 0.5,
        "distance_smoothing": 2.0,
        "min_size": 100,
    },
    "vit_l_lm": {
        "center_distance_thresh": 0.4,
        "boundary_distance_thresh": 0.4,
        "distance_smoothing": 1.6,
        "min_size": 100,
    },
    "vit_h_lm": {
        "center_distance_thresh": 0.5,
        "boundary_distance_thresh": 0.5,
        "distance_smoothing": 1.4,
        "min_size": 100,
    },

    "vit_t_em_organelles": {
        "center_distance_thresh": 0.4,
        "boundary_distance_thresh": 0.5,
        "distance_smoothing": 1.2,
        "min_size": 100,
    },
    "vit_b_em_organelles": {
        "center_distance_thresh": 0.3,
        "boundary_distance_thresh": 0.4,
        "distance_smoothing": 1.2,
        "min_size": 100,
    },
    "vit_l_em_organelles": {
        "center_distance_thresh": 0.3,
        "boundary_distance_thresh": 0.4,
        "distance_smoothing": 1.2,
        "min_size": 100,
    },
    "vit_h_em_organelles": {
        "center_distance_thresh": 0.3,
        "boundary_distance_thresh": 0.4,
        "distance_smoothing": 1.2,
        "min_size": 100,
    }
}

# The settings for the instance segmentation widget with amg.
AMG_SETTINGS = {
    "vit_t_lm": {
        "pred_iou_thresh": 0.6,
        "stability_score_thresh": 0.65,
        "min_object_size": 100,
    },
    "vit_b_lm": {
        "pred_iou_thresh": 0.65,
        "stability_score_thresh": 0.7,
        "min_object_size": 100,
    },
    "vit_l_lm": {
        "pred_iou_thresh": 0.65,
        "stability_score_thresh": 0.73,
        "min_object_size": 100,
    },
    "vit_h_lm": {
        "pred_iou_thresh": 0.65,
        "stability_score_thresh": 0.7,
        "min_object_size": 100,
    },

    "vit_t_em_organelles": {
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.75,
        "min_object_size": 100,
    },
    "vit_b_em_organelles": {
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.75,
        "min_object_size": 100,
    },
    "vit_l_em_organelles": {
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.8,
        "min_object_size": 100,
    },
    "vit_h_em_organelles": {
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.8,
        "min_object_size": 100,
    },
}

# The settings for the nd segment widget.
ND_SEGMENT_SETTINGS = {
    "vit_t_lm": {
        "projection_mode": "box",
        "iou_threshold": 0.8,
        "box_extension": 0.025,
    },
    "vit_b_lm": {
        "projection_mode": "box",
        "iou_threshold": 0.8,
        "box_extension": 0.025,
    },
    "vit_l_lm": {
        "projection_mode": "box",
        "iou_threshold": 0.8,
        "box_extension": 0.025,
    },
    "vit_h_lm": {
        "projection_mode": "box",
        "iou_threshold": 0.8,
        "box_extension": 0.0025,
    },

    "vit_t_em_organelles": {
        "projection_mode": "single_point",
        "iou_threshold": 0.6,
        "box_extension": 0.025,
    },
    "vit_b_em_organelles": {
        "projection_mode": "single_point",
        "iou_threshold": 0.6,
        "box_extension": 0.025,
    },
    "vit_l_em_organelles": {
        "projection_mode": "single_point",
        "iou_threshold": 0.6,
        "box_extension": 0.025,
    },
    "vit_h_em_organelles": {
        "projection_mode": "single_point",
        "iou_threshold": 0.6,
        "box_extension": 0.025,
    }
}


# Defaults per model family and backbone from the 2026-10 parameter searches, see
# `micro_sam.v2.postprocessing.default_postprocessing`.
# 'dense' is for neurite EM volumes, so only the cells models have it.
DEFAULT_POSTPROCESSING = {
    "cells": {
        "hvit_t": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.5, "density_threshold": 30.0, "min_size": 50,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.9,
                },
                "3d": {
                    "foreground_threshold": 0.5, "density_threshold": 80.0, "min_size": 500,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.9,
                },
            },
            "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
        },
        "hvit_s": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.6, "density_threshold": 20.0, "min_size": 25,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.75,
                },
                "3d": {
                    "foreground_threshold": 0.5, "density_threshold": 100.0, "min_size": 200,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.65,
                },
            },
            "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
        },
        "hvit_b": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.4, "density_threshold": 60.0, "min_size": 50,
                    "sigma": 0.15, "n_iter": 100, "dt": 1.0, "foreground_weight": 0.65,
                },
                "3d": {
                    "foreground_threshold": 0.5, "density_threshold": 80.0, "min_size": 200,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.9,
                },
            },
            "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
        },
        "hvit_l": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.5, "density_threshold": 40.0, "min_size": 25,
                    "sigma": 0.15, "n_iter": 800, "dt": 2.0, "foreground_weight": 0.9,
                },
                "3d": {
                    "foreground_threshold": 0.5, "density_threshold": 80.0, "min_size": 500,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.9,
                },
            },
            "dense": {"beta": 0.4, "density_threshold": 3.0, "sigma": 1.0, "n_iter": 100, "dt": 0.5},
        },
    },
    "em_mito": {
        "hvit_s": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.3, "density_threshold": 60.0, "min_size": 500,
                    "sigma": 1.0, "n_iter": 100, "dt": 2.0, "foreground_weight": 0.5,
                },
                "3d": {
                    "foreground_threshold": 0.3, "density_threshold": 80.0, "min_size": 1000,
                    "sigma": 0.09, "n_iter": 50, "dt": 0.5, "foreground_weight": 0.0,
                },
            },
        },
    },
    "em_nuclei": {
        "hvit_s": {
            "sparse": {
                "2d": {
                    "foreground_threshold": 0.4, "density_threshold": 20.0, "min_size": 200,
                    "sigma": 1.0, "n_iter": 50, "dt": 1.0, "foreground_weight": 1.0,
                },
                "3d": {
                    "foreground_threshold": 0.3, "density_threshold": 80.0, "min_size": 2000,
                    "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.65,
                },
            },
        },
    },
}

# Defaults per model family and backbone from the 2026-10 parameter searches, see
# `micro_sam.v2.automatic_prompt_generation.default_prompt_generation`.
DEFAULT_PROMPT_GENERATION_PARAMETERS = {
    "cells": {
        "hvit_t": {
            "2d": {
                "foreground_threshold": 0.9, "candidate_threshold": 2.0, "n_iter": 50, "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.4, "max_overlap": 0.15, "min_size": 25,
                "refinement": "boxes",
            },
            "3d": {
                "foreground_threshold": 0.8, "n_iter": 50, "candidate_threshold": (1.5, 10.0), "sigma": 1.0,
                "min_candidate_size": 1, "score_threshold": 0.7, "max_overlap": 0.3, "min_size": 25,
                "refinement": None,
            },
        },
        "hvit_s": {
            "2d": {
                "foreground_threshold": 0.7, "candidate_threshold": 3.0, "n_iter": 50, "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.4, "max_overlap": 0.3, "min_size": 25,
                "refinement": "boxes",
            },
            "3d": {
                "foreground_threshold": 0.8, "n_iter": 50, "candidate_threshold": (0.5, 5.0), "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.6, "max_overlap": 0.3, "min_size": 25,
                "refinement": None,
            },
        },
        "hvit_b": {
            "2d": {
                "foreground_threshold": 0.6, "candidate_threshold": 2.0, "n_iter": 50, "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.5, "max_overlap": 0.15, "min_size": 25,
                "refinement": "boxes",
            },
            "3d": {
                "foreground_threshold": 0.7, "n_iter": 50, "candidate_threshold": (0.5, 5.0), "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.6, "max_overlap": 0.3, "min_size": 400,
                "refinement": "points+boxes",
            },
        },
        "hvit_l": {
            "2d": {
                "foreground_threshold": 0.7, "candidate_threshold": 2.0, "n_iter": 50, "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.5, "max_overlap": 0.3, "min_size": 15,
                "refinement": "boxes",
            },
            "3d": {
                "foreground_threshold": 0.9, "n_iter": 50, "candidate_threshold": (1.0, 5.0), "sigma": 0.5,
                "min_candidate_size": 1, "score_threshold": 0.7, "max_overlap": 0.3, "min_size": 400,
                "refinement": None,
            },
        },
    },
    "em_mito": {
        "hvit_s": {
            "2d": {
                "foreground_threshold": 0.7, "candidate_threshold": 2.0, "n_iter": 50, "sigma": 0.5,
                "min_candidate_size": 4, "score_threshold": 0.2, "max_overlap": 0.1, "min_size": 200,
                "refinement": None,
            },
            "3d": {
                "foreground_threshold": 0.8, "n_iter": 50, "candidate_threshold": (2.0, 10.0), "sigma": 1.0,
                "min_candidate_size": 1, "score_threshold": 0.7, "max_overlap": 0.15, "min_size": 25,
                "refinement": None,
            },
        },
    },
    "em_nuclei": {
        "hvit_s": {
            "2d": {
                "foreground_threshold": 0.7, "candidate_threshold": 1.0, "n_iter": 50, "sigma": 1.0,
                "min_candidate_size": 4, "score_threshold": 0.7, "max_overlap": 0.5, "min_size": 25,
                "refinement": "boxes",
            },
            "3d": {
                "foreground_threshold": 0.8, "n_iter": 50, "candidate_threshold": (0.5, 5.0), "sigma": 1.0,
                "min_candidate_size": 1, "score_threshold": 0.6, "max_overlap": 0.3, "min_size": 100,
                "refinement": "points+boxes",
            },
        },
    },
}
