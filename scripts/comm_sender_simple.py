#!/usr/bin/env python3
"""
自动变频通讯测量脚本。

默认只需要修改 TARGET_IP：
1. TARGET_IP = SERVER_IP 时，实验标签为 nano_to_server
2. TARGET_IP != SERVER_IP 时，实验标签为 nano_to_nano

运行：
python3 scripts/comm_sender_simple.py
或
python3 scripts/comm_sender_simple.py 192.168.31.105
"""

import csv
import os
import pickle
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.SplitModel import ResNet18_entire


SERVER_IP = "192.168.31.105"
TARGET_IP = "192.168.31.105"
PORT = 8080

TRIALS_PER_MODE = 10
TRIAL_COOLDOWN_SECONDS = 15
MODE_COOLDOWN_SECONDS = 20
SOCKET_TIMEOUT = 120

FALLBACK_MODE_IDS = [0]
OUTPUT_DIR = os.path.join("logs", "comm_energy")


def now_text():
    return datetime.now().isoformat(timespec="microseconds")


def experiment_label(target_ip):
    return "nano_to_server" if target_ip == SERVER_IP else "nano_to_nano"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def latest_log_path(label):
    return os.path.join(OUTPUT_DIR, f"{label}_trials_latest.csv")


def archive_log_path(label):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{label}_trials_{stamp}.csv")


def init_csv(path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment_label",
                "target_ip",
                "target_port",
                "mode",
                "mode_source",
                "trial",
                "start",
                "end",
                "duration_s",
                "bytes",
                "status",
                "error",
            ],
        )
        writer.writeheader()


def append_csv(path, row):
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        writer.writerow(row)


def run_checked(cmd):
    subprocess.run(cmd, check=True)


def set_hardware_mode(mode_id):
    run_checked(["sudo", "nvpmodel", "-m", str(mode_id)])
    run_checked(["sudo", "jetson_clocks"])
    try:
        subprocess.run(
            [
                "bash",
                "-lc",
                "echo 255 | sudo tee "
                "/sys/devices/virtual/thermal/cooling_device3/cur_state >/dev/null",
            ],
            check=True,
        )
    except Exception:  # noqa: BLE001
        pass


def detect_mode_ids():
    # try:
    #     result = subprocess.check_output(
    #         ["sudo", "nvpmodel", "-p", "--verbose"],
    #         stderr=subprocess.STDOUT,
    #     ).decode()
    # except Exception:  # noqa: BLE001
        return list(FALLBACK_MODE_IDS), "mamual"

    # mode_ids = sorted({int(match) for match in re.findall(r"ID:\s*(\d+)", result)})
    # if not mode_ids:
    #     return list(FALLBACK_MODE_IDS), "fallback"
    # return mode_ids, "auto"


def build_payload():
    model = ResNet18_entire()
    model.eval()
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return pickle.dumps(state_dict, protocol=pickle.HIGHEST_PROTOCOL)


def send_once(target_ip, payload):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(SOCKET_TIMEOUT)
        sock.connect((target_ip, PORT))
        sock.sendall(len(payload).to_bytes(8, byteorder="big"))
        sock.sendall(payload)
        ack = sock.recv(2)
        if ack != b"ok":
            raise RuntimeError(f"bad ack: {ack!r}")


def main():
    target_ip = sys.argv[1] if len(sys.argv) > 1 else TARGET_IP
    label = experiment_label(target_ip)

    ensure_output_dir()
    latest_path = latest_log_path(label)
    archive_path = archive_log_path(label)
    init_csv(latest_path)
    init_csv(archive_path)

    payload = build_payload()
    payload_bytes = len(payload)
    modes, mode_source = detect_mode_ids()

    print(
        f"experiment={label} target={target_ip}:{PORT} bytes={payload_bytes} "
        f"modes={modes} mode_source={mode_source}",
        flush=True,
    )
    print(f"log_latest={latest_path}", flush=True)
    print(f"log_archive={archive_path}", flush=True)

    for mode_index, mode_id in enumerate(modes, start=1):
        print(f"[mode {mode_index}/{len(modes)}] switching to {mode_id}", flush=True)
        set_hardware_mode(mode_id)
        time.sleep(MODE_COOLDOWN_SECONDS)

        for trial_idx in range(1, TRIALS_PER_MODE + 1):
            start_wall = time.time()
            start_time = now_text()
            status = "ok"
            error = ""

            try:
                send_once(target_ip, payload)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = str(exc)

            end_time = now_text()
            duration = time.time() - start_wall
            row = {
                "experiment_label": label,
                "target_ip": target_ip,
                "target_port": PORT,
                "mode": mode_id,
                "mode_source": mode_source,
                "trial": trial_idx,
                "start": start_time,
                "end": end_time,
                "duration_s": f"{duration:.6f}",
                "bytes": payload_bytes,
                "status": status,
                "error": error,
            }
            append_csv(latest_path, row)
            append_csv(archive_path, row)
            print(
                f"mode={mode_id} trial={trial_idx}/{TRIALS_PER_MODE} "
                f"duration={duration:.6f}s status={status}",
                flush=True,
            )

            if trial_idx < TRIALS_PER_MODE:
                time.sleep(TRIAL_COOLDOWN_SECONDS)

        if mode_index < len(modes):
            time.sleep(MODE_COOLDOWN_SECONDS)


if __name__ == "__main__":
    main()
