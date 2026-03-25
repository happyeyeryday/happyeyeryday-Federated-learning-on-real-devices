MODEL_IDX_TO_DEPTH_RATIO = {
    1: 0.25,
    2: 0.5,
    3: 0.75,
    4: 1.0,
}

CID_TO_DEVICE_TYPE = {
    0: "nano",
    1: "nano",
    2: "nano",
    3: "nano",
    4: "nano",
    5: "nano",
    6: "nano",
    7: "agx_xavier",
    8: "agx_xavier",
    9: "orinnanosuper",
}

DEVICE_MAX_MODEL_IDX = {
    "nano": 2,
    "agx_xavier": 3,
    "orinnanosuper": 4,
}

DEVICE_DVFS_MODES = {
    "nano": {"low": 1, "high": 0},
    "agx_xavier": {"low": 2, "high": 0},
    "orinnanosuper": {"low": 0, "high": 2},
}

DEVICE_COST_SCORE = {
    "nano": 1.0,
    "agx_xavier": 0.55,
    "orinnanosuper": 0.35,
}


def get_device_type(cid):
    return CID_TO_DEVICE_TYPE.get(int(cid), "nano")


def model_depth_ratio_from_idx(model_idx):
    return float(MODEL_IDX_TO_DEPTH_RATIO[int(model_idx)])


def max_model_idx_for_device(device_type):
    return int(DEVICE_MAX_MODEL_IDX[device_type])


def dvfs_mode_for(device_type, label):
    return int(DEVICE_DVFS_MODES[device_type][label])


def estimated_cost(device_type, model_idx):
    depth_ratio = model_depth_ratio_from_idx(model_idx)
    return depth_ratio / DEVICE_COST_SCORE[device_type]


def choose_model_idx(device_type, appearance_count):
    model_idx = max_model_idx_for_device(device_type)
    if appearance_count >= 8 and model_idx > 1:
        model_idx -= 1
    return model_idx


def choose_dvfs_label(model_idx, selected_rank):
    if int(model_idx) >= 3:
        return "high"
    if int(selected_rank) == 0:
        return "high"
    return "low"
