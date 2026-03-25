from pathlib import Path


LOCAL_CONFIG = Path(__file__).with_name("devices_local.py")
EXAMPLE_CONFIG = Path(__file__).with_name("devices_example.py")


def load_device_lists():
    namespace = {}
    config_path = LOCAL_CONFIG if LOCAL_CONFIG.exists() else EXAMPLE_CONFIG
    if not config_path.exists():
        raise RuntimeError(
            "Missing deploy device config. Create deploy_sh/devices_local.py first."
        )

    exec(config_path.read_text(encoding="utf-8"), namespace)
    return (
        namespace.get("SERVER_DEVICES", []),
        namespace.get("CLIENT_DEVICES", []),
        namespace.get("ALL_DEVICES", []),
    )


SERVER_DEVICES, CLIENT_DEVICES, ALL_DEVICES = load_device_lists()
