MODEL_ID_TO_RATE = {
    0: 0.125,
    1: 0.25,
    2: 0.5,
    3: 1.0,
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

DEVICE_MAX_MODEL_ID = {
    "nano": 1,
    "agx_xavier": 2,
    "orinnanosuper": 3,
}

# Real-device DVFS modes are intentionally isolated here for easy later edits.
DEVICE_DVFS_MODES = {
    "nano": {
        "low": 0,
        "high": 1,
    },
    "agx_xavier": {
        "low": 0,
        "high": 1,
    },
    "orinnanosuper": {
        "low": 0,
        "high": 1,
    },
}

# Lower score means weaker device and higher estimated runtime cost.
DEVICE_COST_SCORE = {
    "nano": 1.0,
    "agx_xavier": 0.55,
    "orinnanosuper": 0.35,
}


def get_device_type(cid: int) -> str:
    return CID_TO_DEVICE_TYPE.get(int(cid), "nano")


def model_rate_from_id(model_id: int) -> float:
    return MODEL_ID_TO_RATE[int(model_id)]


def max_model_id_for_device(device_type: str) -> int:
    return DEVICE_MAX_MODEL_ID[device_type]


def dvfs_mode_for(device_type: str, label: str) -> int:
    return int(DEVICE_DVFS_MODES[device_type][label])


def estimated_cost(device_type: str, model_id: int) -> float:
    # Larger model on weaker device gets a higher cost.
    width = MODEL_ID_TO_RATE[int(model_id)]
    return width / DEVICE_COST_SCORE[device_type]


def choose_model_id(device_type: str, appearance_count: int) -> int:
    max_model_id = max_model_id_for_device(device_type)
    if appearance_count >= 8 and max_model_id > 0:
        return max_model_id - 1
    return max_model_id


def choose_dvfs_label(model_id: int, selected_rank: int) -> str:
    if int(model_id) >= 2:
        return "high"
    if selected_rank == 0:
        return "high"
    return "low"
