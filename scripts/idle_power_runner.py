#!/usr/bin/env python3
"""
静息功耗测量脚本。

流程：
1. 切换到指定 nvpmodel 模式并锁定频率
2. 等待 MODE_STABILIZE_SECONDS，让频率和温度稳定
3. 在空载状态下记录静息测量时间窗

注意：
- 本脚本不直接读取功率计，只负责记录静息功耗的测量时间段。
- 后续可按 CSV 中的 start_time / end_time 去匹配功率计导出的 xlsx 数据。
"""

import csv
import os
import subprocess
import time
from datetime import datetime


# ================= 配置区域 =================
# 你可以根据设备手动修改这里的列表
# Orin 建议: [0, 1, 2, 3] (对应 MAXN, 15W, 30W, 50W)
# Xavier 建议: [0, 2, 3]
# Nano 建议: [0, 1]
TARGET_MODES = [0, 1, 2]

MODE_STABILIZE_SECONDS = 30
MEASURE_SECONDS = 60
MODE_COOLDOWN_SECONDS = 20

OUTPUT_DIR = os.path.join("logs", "idle_energy")
LOG_FILE = os.path.join(OUTPUT_DIR, "idle_power_latest.csv")
# ===========================================


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_csv():
    with open(LOG_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "device_model",
                "mode",
                "stabilize_seconds",
                "measure_seconds",
                "start_time",
                "end_time",
                "duration_s",
                "status",
                "error",
            ]
        )


def append_row(row):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([sanitize_csv_value(value) for value in row])


def sanitize_csv_value(value):
    text = str(value)
    return text.replace("\x00", "").replace("\r", " ").replace("\n", " ").strip()


def set_jetson_hardware(mode_id):
    print(f"\n>>> 正在切换到模式 ID: {mode_id}", flush=True)
    result = subprocess.run(
        ["sudo", "nvpmodel", "-m", str(mode_id)],
        input="NO\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output = result.stdout + result.stderr
    if "Reboot required" in output or "REBOOT" in output.upper():
        print(f"[跳过] 模式 {mode_id} 需要重启才能切换，本次跳过。", flush=True)
        print(f"       若要测试该模式，请手动执行: sudo nvpmodel -m {mode_id} 并重启后再运行脚本。", flush=True)
        return False, "reboot required"
    if result.returncode != 0:
        raise RuntimeError(output.strip() or f"nvpmodel failed for mode {mode_id}")

    subprocess.run(["sudo", "jetson_clocks"], check=True)
    try:
        subprocess.run(
            "echo 255 | sudo tee /sys/devices/virtual/thermal/cooling_device3/cur_state",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except Exception:
        pass

    print(f">>> 模式 {mode_id} 已锁定，风扇已开启。", flush=True)
    return True, ""


def measure_idle_window(mode_id):
    print(f"  等待模式稳定 {MODE_STABILIZE_SECONDS} 秒...", flush=True)
    time.sleep(MODE_STABILIZE_SECONDS)

    start_time_str = now_text()
    t_start = time.time()
    print(f"  ▶ Mode {mode_id} 静息测量开始: {start_time_str}", flush=True)

    time.sleep(MEASURE_SECONDS)

    t_end = time.time()
    end_time_str = now_text()
    duration_s = t_end - t_start
    print(
        f"  ■ Mode {mode_id} 静息测量结束: {end_time_str}  (静息历时 {duration_s:.1f} 秒)",
        flush=True,
    )
    return start_time_str, end_time_str, duration_s


def read_device_model():
    try:
        with open("/proc/device-tree/model", "r") as fh:
            return sanitize_csv_value(fh.read())
    except Exception:
        return "unknown"


def main():
    ensure_output_dir()
    init_csv()
    device_model = read_device_model()

    print(f"device_model={device_model}", flush=True)
    print(f"log_file={LOG_FILE}", flush=True)

    for mode_index, mode_id in enumerate(TARGET_MODES, start=1):
        print(f"\n{'=' * 60}", flush=True)
        print(f"[Mode {mode_index}/{len(TARGET_MODES)}] 开始静息功耗测量", flush=True)

        start_time = ""
        end_time = ""
        duration_s = 0.0
        status = "ok"
        error = ""

        try:
            switched, switch_error = set_jetson_hardware(mode_id)
            if not switched:
                status = "skipped"
                error = switch_error
            else:
                start_time, end_time, duration_s = measure_idle_window(mode_id)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            print(f"[失败] mode {mode_id}: {error}", flush=True)

        append_row(
            [
                device_model,
                mode_id,
                MODE_STABILIZE_SECONDS,
                MEASURE_SECONDS,
                start_time,
                end_time,
                f"{duration_s:.2f}",
                status,
                error,
            ]
        )

        print(f"[Mode {mode_id}] status={status}", flush=True)
        if error:
            print(f"[Mode {mode_id}] error={error}", flush=True)

        if mode_index < len(TARGET_MODES):
            print(f"[Mode {mode_id}] 冷却 {MODE_COOLDOWN_SECONDS} 秒后进入下一个模式...", flush=True)
            time.sleep(MODE_COOLDOWN_SECONDS)


if __name__ == "__main__":
    main()
