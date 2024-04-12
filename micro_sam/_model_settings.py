# FIXME all current values are dummy values just for checking that the updates work.
# replace with the real values for vit_t_lm.

# The settings for the instance segmentation widget with ais.
AIS_SETTINGS = {
    "vit_t_lm": {
        "center_distance_thresh": 0.25,
        "boundary_distance_thresh": 0.35,
    }
}

# The settings for the instance segmentation widget with amg.
AMG_SETTINGS = {
    "vit_t_lm": {
        "pred_iou_thresh": 0.6,
        "stability_score_thresh": 0.55,
        "min_object_size": 125,
    }
}

# The settings for the nd segment widget.
ND_SEGMENT_SETTINGS = {
    "vit_t_lm": {
        "projection_mode": "points",
        "iou_threshold": 0.55,
        "box_extension": 0.01,
    }
}
