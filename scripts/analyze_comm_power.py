#!/usr/bin/env python3
"""
通讯功耗分析脚本。

默认输入：
1. scripts/nano/nanocomm.csv
2. scripts/nano/nano.xlsx

支持命令行覆盖：
python3 scripts/analyze_comm_power.py <comm_log.csv> <power.xlsx>
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from xml.etree import ElementTree as ET
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

DEFAULT_INPUT_DIR = BASE_DIR / "nano"
COMM_LOG_DIR = REPO_ROOT / "logs" / "comm_energy"
DEFAULT_COMM_LOG_PATH = DEFAULT_INPUT_DIR / "nanocomm.csv"
POWER_XLSX_PATH = DEFAULT_INPUT_DIR / "nano.xlsx"

TIME_COLUMN = "A"
POWER_COLUMN = "E"
EXCEL_EPOCH = datetime(1899, 12, 30)

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def default_comm_log_path():
    if DEFAULT_COMM_LOG_PATH.exists():
        return DEFAULT_COMM_LOG_PATH
    candidates = sorted(COMM_LOG_DIR.glob("*_trials_latest.csv"), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]
    return COMM_LOG_DIR / "nano_to_server_trials_latest.csv"


def output_paths(comm_log_path):
    out_dir = comm_log_path.resolve().parent
    return (
        out_dir / "comm_power_result.csv",
        out_dir / "comm_power_trials.csv",
        out_dir / "comm_power_summary.txt",
    )


def excel_time_to_datetime(value):
    return EXCEL_EPOCH + timedelta(days=float(value))


def read_shared_strings(zip_file):
    shared = []
    name = "xl/sharedStrings.xml"
    if name not in zip_file.namelist():
        return shared

    root = ET.fromstring(zip_file.read(name))
    for si in root:
        text = "".join(node.text or "" for node in si.iter(NS_MAIN + "t"))
        shared.append(text)
    return shared


def first_sheet_name(zip_file):
    workbook_root = ET.fromstring(zip_file.read("xl/workbook.xml"))
    sheet_node = workbook_root.find(f"{NS_MAIN}sheets/{NS_MAIN}sheet")
    if sheet_node is None:
        raise ValueError("no worksheet found in workbook.xml")

    relation_id = sheet_node.attrib.get(NS_REL + "id")
    if not relation_id:
        raise ValueError("worksheet relationship id missing")

    rels_root = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    for rel in rels_root:
        if rel.attrib.get("Id") == relation_id:
            target = rel.attrib.get("Target", "")
            normalized = target.lstrip("/")
            if normalized in zip_file.namelist():
                return normalized
            if normalized.startswith("xl/"):
                return normalized
            return "xl/" + normalized
    raise ValueError(f"worksheet target missing for relation {relation_id}")


def cell_text(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    value_node = cell.find(NS_MAIN + "v")
    inline_node = cell.find(NS_MAIN + "is")

    if cell_type == "inlineStr" and inline_node is not None:
        return "".join(node.text or "" for node in inline_node.iter(NS_MAIN + "t"))

    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell_type == "s" and value:
        return shared_strings[int(value)]
    return value


def load_power_samples(xlsx_path):
    with ZipFile(xlsx_path) as zip_file:
        shared_strings = read_shared_strings(zip_file)
        sheet_name = first_sheet_name(zip_file)
        sheet_root = ET.fromstring(zip_file.read(sheet_name))
        sheet_data = sheet_root.find(NS_MAIN + "sheetData")
        if sheet_data is None:
            raise ValueError("sheetData missing in worksheet")

        samples = []
        for row_index, row in enumerate(sheet_data, start=1):
            values = {}
            for cell in row:
                ref = cell.attrib.get("r", "")
                column = "".join(ch for ch in ref if ch.isalpha())
                if not column:
                    continue
                values[column] = cell_text(cell, shared_strings)

            if row_index == 1:
                continue
            if TIME_COLUMN not in values or POWER_COLUMN not in values:
                continue

            samples.append(
                {
                    "time": excel_time_to_datetime(values[TIME_COLUMN]),
                    "power_w": float(values[POWER_COLUMN]),
                }
            )

    if not samples:
        raise ValueError(f"no power samples loaded from {xlsx_path}")
    return samples


def load_trials(comm_log_path):
    trials = []
    with comm_log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status") != "ok":
                continue
            trials.append(
                {
                    "experiment_label": row["experiment_label"],
                    "target_ip": row["target_ip"],
                    "target_port": int(row["target_port"]),
                    "mode": int(row["mode"]),
                    "mode_source": row["mode_source"],
                    "trial": int(row["trial"]),
                    "start": datetime.fromisoformat(row["start"]),
                    "end": datetime.fromisoformat(row["end"]),
                    "duration_s": float(row["duration_s"]),
                    "bytes": int(row["bytes"]),
                }
            )

    if not trials:
        raise ValueError(f"no successful trials found in {comm_log_path}")
    return trials


def validate_single_experiment(trials):
    experiment_keys = {
        (trial["experiment_label"], trial["target_ip"], trial["target_port"]) for trial in trials
    }
    if len(experiment_keys) != 1:
        raise ValueError(
            "comm log must contain exactly one experiment_label/target_ip/target_port combination"
        )


def in_any_trial(timestamp, trials):
    for trial in trials:
        if trial["start"] <= timestamp <= trial["end"]:
            return True
    return False


def estimate_baseline(samples, trials):
    window_start = min(trial["start"] for trial in trials)
    window_end = max(trial["end"] for trial in trials)
    local_idle_samples = [
        sample["power_w"]
        for sample in samples
        if window_start <= sample["time"] <= window_end and not in_any_trial(sample["time"], trials)
    ]
    if local_idle_samples:
        return mean(local_idle_samples)

    idle_samples = [sample["power_w"] for sample in samples if not in_any_trial(sample["time"], trials)]
    if idle_samples:
        return mean(idle_samples)
    return mean(sample["power_w"] for sample in samples)


def analyze_trial(trial, samples, baseline_power):
    matched = [sample for sample in samples if trial["start"] <= sample["time"] <= trial["end"]]
    if not matched:
        return None

    avg_power = mean(sample["power_w"] for sample in matched)
    gross_energy = avg_power * trial["duration_s"]
    net_energy = max(avg_power - baseline_power, 0.0) * trial["duration_s"]

    result = dict(trial)
    result["sample_count"] = len(matched)
    result["avg_power_w"] = avg_power
    result["gross_energy_j"] = gross_energy
    result["net_energy_j"] = net_energy
    return result


def write_trial_csv(rows, output_trial_csv):
    with output_trial_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
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
                "sample_count",
                "avg_power_w",
                "gross_energy_j",
                "net_energy_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["experiment_label"],
                    row["target_ip"],
                    row["target_port"],
                    row["mode"],
                    row["mode_source"],
                    row["trial"],
                    row["start"].isoformat(),
                    row["end"].isoformat(),
                    f'{row["duration_s"]:.6f}',
                    row["bytes"],
                    row["sample_count"],
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["gross_energy_j"]:.6f}',
                    f'{row["net_energy_j"]:.6f}',
                ]
            )


def summarize_modes(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    summaries = []
    for mode in sorted(grouped):
        items = grouped[mode]
        duration_values = [item["duration_s"] for item in items]
        summary = {
            "experiment_label": items[0]["experiment_label"],
            "target_ip": items[0]["target_ip"],
            "target_port": items[0]["target_port"],
            "mode": mode,
            "mode_source": items[0]["mode_source"],
            "trial_count": len(items),
            "bytes": items[0]["bytes"],
            "avg_duration_s": mean(duration_values),
            "std_duration_s": stdev(duration_values) if len(duration_values) > 1 else 0.0,
            "avg_power_w": mean(item["avg_power_w"] for item in items),
            "avg_gross_energy_j": mean(item["gross_energy_j"] for item in items),
            "avg_net_energy_j": mean(item["net_energy_j"] for item in items),
        }
        summaries.append(summary)
    return summaries


def write_mode_csv(rows, output_mode_csv):
    with output_mode_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "experiment_label",
                "target_ip",
                "target_port",
                "mode",
                "mode_source",
                "trial_count",
                "bytes",
                "avg_duration_s",
                "std_duration_s",
                "avg_power_w",
                "avg_gross_energy_j",
                "avg_net_energy_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["experiment_label"],
                    row["target_ip"],
                    row["target_port"],
                    row["mode"],
                    row["mode_source"],
                    row["trial_count"],
                    row["bytes"],
                    f'{row["avg_duration_s"]:.6f}',
                    f'{row["std_duration_s"]:.6f}',
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["avg_gross_energy_j"]:.6f}',
                    f'{row["avg_net_energy_j"]:.6f}',
                ]
            )


def make_summary(mode_rows, baseline_power):
    if not mode_rows:
        return "no mode rows"

    label = mode_rows[0]["experiment_label"]
    target = f'{mode_rows[0]["target_ip"]}:{mode_rows[0]["target_port"]}'

    fastest = min(mode_rows, key=lambda row: row["avg_duration_s"])
    lowest_net = min(mode_rows, key=lambda row: row["avg_net_energy_j"])

    lines = [
        f"experiment_label = {label}",
        f"target = {target}",
        f"baseline_power_w = {baseline_power:.6f}",
        "",
        "per_mode",
    ]

    for row in mode_rows:
        lines.append(
            "  mode={mode} trial_count={trial_count} avg_duration_s={avg_duration_s:.6f} "
            "std_duration_s={std_duration_s:.6f} avg_power_w={avg_power_w:.6f} "
            "avg_gross_energy_j={avg_gross_energy_j:.6f} avg_net_energy_j={avg_net_energy_j:.6f}".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "best",
            f"  fastest_mode = {fastest['mode']} ({fastest['avg_duration_s']:.6f} s)",
            f"  lowest_net_energy_mode = {lowest_net['mode']} ({lowest_net['avg_net_energy_j']:.6f} J)",
        ]
    )
    return "\n".join(lines)


def main():
    comm_log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_comm_log_path()
    power_xlsx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else POWER_XLSX_PATH
    output_mode_csv, output_trial_csv, output_summary = output_paths(comm_log_path)

    samples = load_power_samples(power_xlsx_path)
    trials = load_trials(comm_log_path)
    validate_single_experiment(trials)
    baseline_power = estimate_baseline(samples, trials)

    analyzed_trials = []
    for trial in trials:
        result = analyze_trial(trial, samples, baseline_power)
        if result is not None:
            analyzed_trials.append(result)

    if not analyzed_trials:
        raise ValueError("no trials matched any power samples")

    mode_rows = summarize_modes(analyzed_trials)
    write_trial_csv(analyzed_trials, output_trial_csv)
    write_mode_csv(mode_rows, output_mode_csv)

    summary = make_summary(mode_rows, baseline_power)
    output_summary.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"trial csv written to: {output_trial_csv}")
    print(f"mode csv written to: {output_mode_csv}")
    print(f"summary written to: {output_summary}")


if __name__ == "__main__":
    main()
