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
