from typing import Optional

from utils.helcfl_real_profiles import choose_dvfs_label as helcfl_choose_dvfs_label


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

DEVICE_STRENGTH = {
    "nano": 1.0,
    "agx_xavier": 1.8,
    "orinnanosuper": 2.6,
}

BASE_MODEL_IDX = {
    "nano": 2,
    "agx_xavier": 3,
    "orinnanosuper": 4,
}

COOLDOWN_ROUNDS = 1
FAILURE_TOLERANCE = 2
FAIRNESS_ALPHA = 1.0
LAG_TOLERANCE_SEC = 300.0


def get_device_type(cid):
    return CID_TO_DEVICE_TYPE.get(int(cid), "nano")


def device_strength(device_type):
    return float(DEVICE_STRENGTH[device_type])


def max_model_idx_for_device(device_type):
    return int(DEVICE_MAX_MODEL_IDX[device_type])


def model_depth_ratio_from_idx(model_idx):
    return float(MODEL_IDX_TO_DEPTH_RATIO[int(model_idx)])


def dvfs_mode_for(device_type, label):
    return int(DEVICE_DVFS_MODES[device_type][label])


def fairness_penalty(weighted_participation, mean_weighted_participation, alpha=FAIRNESS_ALPHA):
    overload = max(0.0, float(weighted_participation) - float(mean_weighted_participation) + 1.0)
    return max(1.0, overload ** alpha)


def choose_model_idx(
    device_type,
    appearance_count,
    failure_count,
    lagged_last_round=False,
    last_model_idx: Optional[int] = None,
):
    model_idx = int(BASE_MODEL_IDX[device_type])

    if appearance_count >= 8:
        model_idx -= 1
    if failure_count >= 1:
        model_idx -= 1
    if lagged_last_round:
        model_idx -= 1

    if failure_count >= 2 and last_model_idx is not None:
        model_idx = min(model_idx, int(last_model_idx))

    return max(1, min(model_idx, max_model_idx_for_device(device_type)))


def choose_dvfs_label(model_idx, selected_rank):
    # Keep CAMA aligned with HELCFL on DVFS so the algorithmic difference
    # stays in client selection and model_idx assignment.
    return helcfl_choose_dvfs_label(model_idx, selected_rank)


def is_on_cooldown(last_selected_round, round_idx, cooldown_rounds=COOLDOWN_ROUNDS):
    if last_selected_round is None:
        return False
    return (int(round_idx) - int(last_selected_round)) <= int(cooldown_rounds)


def cama_utility(
    device_type,
    model_idx,
    weighted_participation,
    mean_weighted_participation,
    alpha=FAIRNESS_ALPHA,
):
    depth_ratio = model_depth_ratio_from_idx(model_idx)
    capability = depth_ratio * device_strength(device_type)
    penalty = fairness_penalty(weighted_participation, mean_weighted_participation, alpha)
    return capability / penalty
