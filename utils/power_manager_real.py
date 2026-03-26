from loguru import logger


POWER_CONFIG = {
    "nano": {
        "low": {
            "idle": 2.838381,
            "communication": 2.975577,
            "train": 8.5305,
        },
        "high": {
            "idle": 3.038518,
            "communication": 3.215376,
            "train": 6.5203,
        },
    },
    "agx_xavier": {
        "low": {
            "idle": 7.504458,
            "communication": 7.613694,
            "train": 15.3907,
        },
        "high": {
            "idle": 9.642757,
            "communication": 9.895526,
            "train": 31.3770,
        },
    },
    "orinnanosuper": {
        "low": {
            "idle": 8.426972,
            "communication": 8.547448,
            "train": 16.8472,
        },
        "high": {
            "idle": 9.199202,
            "communication": 9.324621,
            "train": 21.2785,
        },
    },
}

BATTERY_CAPACITY = {
    "nano": 150000.0,
    "agx_xavier": 150000.0,
    "orinnanosuper": 150000.0,
}

DEVICE_DVFS_MODES = {
    "nano": {"low": 1, "high": 0},
    "agx_xavier": {"low": 2, "high": 0},
    "orinnanosuper": {"low": 0, "high": 2},
}

LOW_BATTERY_THRESHOLD_J = 50.0


def get_device_capacity(device_type):
    return float(BATTERY_CAPACITY[device_type])


def normalize_battery(device_type, joules):
    capacity = get_device_capacity(device_type)
    return min(max(float(joules), 0.0), capacity) / capacity


def mode_label_from_id(device_type, mode_id):
    mode_id = int(mode_id)
    for label, configured_id in DEVICE_DVFS_MODES[device_type].items():
        if int(configured_id) == mode_id:
            return label
    raise ValueError(f"Unknown mode id {mode_id} for device type {device_type}")


class BatteryManagerReal:
    def __init__(self, device_type, initial_mode_label="low"):
        if device_type not in POWER_CONFIG:
            raise ValueError(f"Unknown device_type: {device_type}")
        self.device_type = device_type
        self.total_capacity = get_device_capacity(device_type)
        self.current_charge = float(self.total_capacity)
        self.current_mode_label = str(initial_mode_label)
        logger.info(
            f"[BatteryReal] Initialized device_type={device_type} "
            f"mode={self.current_mode_label} capacity={self.total_capacity:.1f}J"
        )

    def set_charge(self, charge_joules):
        self.current_charge = min(max(float(charge_joules), 0.0), self.total_capacity)
        logger.info(
            f"[BatteryReal] Restored charge: "
            f"{self.current_charge:.2f}/{self.total_capacity:.2f}J "
            f"({self.get_ratio() * 100:.2f}%)"
        )
        return self.current_charge

    def get_charge(self):
        return float(self.current_charge)

    def get_ratio(self):
        return self.current_charge / self.total_capacity

    def set_power_mode(self, mode_label):
        if mode_label not in POWER_CONFIG[self.device_type]:
            raise ValueError(f"Unknown mode label {mode_label} for {self.device_type}")
        self.current_mode_label = str(mode_label)
        logger.info(f"[BatteryReal] Mode label set to {self.current_mode_label}")

    def set_power_mode_by_mode_id(self, mode_id):
        mode_label = mode_label_from_id(self.device_type, mode_id)
        self.set_power_mode(mode_label)
        return mode_label

    def consume(self, activity, duration_seconds, mode_label=None):
        activity = str(activity)
        if activity not in ("idle", "communication", "train"):
            logger.warning(f"Unknown activity '{activity}', falling back to 'idle'")
            activity = "idle"

        duration_seconds = max(0.0, float(duration_seconds))
        if duration_seconds == 0.0:
            return self.current_charge

        active_mode = mode_label or self.current_mode_label
        if active_mode not in POWER_CONFIG[self.device_type]:
            raise ValueError(f"Unknown mode label {active_mode} for {self.device_type}")

        power_w = POWER_CONFIG[self.device_type][active_mode][activity]
        consumed_joules = power_w * duration_seconds
        self.current_charge = max(0.0, self.current_charge - consumed_joules)

        logger.info(
            f"[BatteryReal] activity={activity} mode={active_mode} "
            f"duration={duration_seconds:.3f}s power={power_w:.4f}W "
            f"consumed={consumed_joules:.4f}J "
            f"remaining={self.current_charge:.2f}/{self.total_capacity:.2f}J "
            f"({self.get_ratio() * 100:.2f}%)"
        )
        return self.current_charge

    def check_energy(self, threshold=LOW_BATTERY_THRESHOLD_J):
        threshold = float(threshold)
        is_enough = self.current_charge > threshold
        if not is_enough:
            logger.warning(
                f"[BatteryReal] Low battery: "
                f"{self.current_charge:.2f}J <= {threshold:.2f}J"
            )
        return is_enough
