#!/usr/bin/env python3
"""
训练功耗分析脚本。

默认输入：
1. scripts/orinnano/orinnanotrain.csv
2. scripts/orinnano/orinnano.xlsx

支持命令行覆盖：
python3 scripts/power_analyzer.py <train_log.csv> <power.xlsx>
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree as ET
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "orinnano"
DEFAULT_TRAIN_LOG = DEFAULT_INPUT_DIR / "orinnanotrain.csv"
DEFAULT_POWER_XLSX = DEFAULT_INPUT_DIR / "orinnano.xlsx"

TIME_COLUMN = "A"
POWER_COLUMN = "E"
EXCEL_EPOCH = datetime(1899, 12, 30)

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def output_paths(train_log_path):
    out_dir = train_log_path.resolve().parent
    return (
        out_dir / "train_power_detail.csv",
        out_dir / "train_power_result.csv",
        out_dir / "train_power_per_batch_result.csv",
        out_dir / "train_power_summary.txt",
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


def load_runs(train_log_path):
    runs = []
    with train_log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            runs.append(
                {
                    "device_model": row["device_model"],
                    "mode": int(row["mode"]),
                    "model_idx": int(row["model_idx"]),
                    "start": datetime.strptime(row["start_time"], "%Y-%m-%d %H:%M:%S"),
                    "end": datetime.strptime(row["end_time"], "%Y-%m-%d %H:%M:%S"),
                    "avg_batch_time_ms": float(row["avg_batch_time_ms"]),
                }
            )

    if not runs:
        raise ValueError(f"no runs found in {train_log_path}")
    return runs


def in_any_run(timestamp, runs):
    for run in runs:
        if run["start"] <= timestamp <= run["end"]:
            return True
    return False


def estimate_baseline(samples, runs):
    window_start = min(run["start"] for run in runs)
    window_end = max(run["end"] for run in runs)
    local_idle_samples = [
        sample["power_w"]
        for sample in samples
        if window_start <= sample["time"] <= window_end and not in_any_run(sample["time"], runs)
    ]
    if local_idle_samples:
        return mean(local_idle_samples)

    idle_samples = [sample["power_w"] for sample in samples if not in_any_run(sample["time"], runs)]
    if idle_samples:
        return mean(idle_samples)
    return mean(sample["power_w"] for sample in samples)


def analyze_run(run, samples, baseline_power):
    matched = [sample for sample in samples if run["start"] <= sample["time"] <= run["end"]]
    if not matched:
        return None

    duration_s = (run["end"] - run["start"]).total_seconds()
    avg_power = mean(sample["power_w"] for sample in matched)
    peak_power = max(sample["power_w"] for sample in matched)
    gross_energy = avg_power * duration_s
    net_energy = max(avg_power - baseline_power, 0.0) * duration_s
    gross_energy_per_batch = avg_power * (run["avg_batch_time_ms"] / 1000.0)
    net_energy_per_batch = max(avg_power - baseline_power, 0.0) * (run["avg_batch_time_ms"] / 1000.0)

    result = dict(run)
    result["duration_s"] = duration_s
    result["sample_count"] = len(matched)
    result["avg_power_w"] = avg_power
    result["peak_power_w"] = peak_power
    result["gross_energy_j"] = gross_energy
    result["net_energy_j"] = net_energy
    result["gross_energy_per_batch_j"] = gross_energy_per_batch
    result["net_energy_per_batch_j"] = net_energy_per_batch
    return result


def summarize_mode_model(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["model_idx"]), []).append(row)

    summaries = []
    for mode, model_idx in sorted(grouped):
        items = grouped[(mode, model_idx)]
        summaries.append(
            {
                "device_model": items[0]["device_model"],
                "mode": mode,
                "model_idx": model_idx,
                "run_count": len(items),
                "avg_duration_s": mean(item["duration_s"] for item in items),
                "avg_batch_time_ms": mean(item["avg_batch_time_ms"] for item in items),
                "avg_power_w": mean(item["avg_power_w"] for item in items),
                "peak_power_w": max(item["peak_power_w"] for item in items),
                "avg_gross_energy_j": mean(item["gross_energy_j"] for item in items),
                "avg_net_energy_j": mean(item["net_energy_j"] for item in items),
                "avg_gross_energy_per_batch_j": mean(
                    item["gross_energy_per_batch_j"] for item in items
                ),
                "avg_net_energy_per_batch_j": mean(
                    item["net_energy_per_batch_j"] for item in items
                ),
            }
        )
    return summaries


def write_detail_csv(rows, output_detail_csv):
    with output_detail_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "device_model",
                "mode",
                "model_idx",
                "start",
                "end",
                "duration_s",
                "avg_batch_time_ms",
                "sample_count",
                "avg_power_w",
                "peak_power_w",
                "gross_energy_j",
                "net_energy_j",
                "gross_energy_per_batch_j",
                "net_energy_per_batch_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_model"],
                    row["mode"],
                    row["model_idx"],
                    row["start"].isoformat(sep=" "),
                    row["end"].isoformat(sep=" "),
                    f'{row["duration_s"]:.6f}',
                    f'{row["avg_batch_time_ms"]:.6f}',
                    row["sample_count"],
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["peak_power_w"]:.6f}',
                    f'{row["gross_energy_j"]:.6f}',
                    f'{row["net_energy_j"]:.6f}',
                    f'{row["gross_energy_per_batch_j"]:.6f}',
                    f'{row["net_energy_per_batch_j"]:.6f}',
                ]
            )


def write_summary_csv(rows, output_summary_csv):
    with output_summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "device_model",
                "mode",
                "model_idx",
                "run_count",
                "avg_duration_s",
                "avg_batch_time_ms",
                "avg_power_w",
                "peak_power_w",
                "avg_gross_energy_j",
                "avg_net_energy_j",
                "avg_gross_energy_per_batch_j",
                "avg_net_energy_per_batch_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_model"],
                    row["mode"],
                    row["model_idx"],
                    row["run_count"],
                    f'{row["avg_duration_s"]:.6f}',
                    f'{row["avg_batch_time_ms"]:.6f}',
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["peak_power_w"]:.6f}',
                    f'{row["avg_gross_energy_j"]:.6f}',
                    f'{row["avg_net_energy_j"]:.6f}',
                    f'{row["avg_gross_energy_per_batch_j"]:.6f}',
                    f'{row["avg_net_energy_per_batch_j"]:.6f}',
                ]
            )


def write_per_batch_csv(rows, output_per_batch_csv):
    with output_per_batch_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "device_model",
                "mode",
                "model_idx",
                "run_count",
                "avg_batch_time_ms",
                "avg_gross_energy_per_batch_j",
                "avg_net_energy_per_batch_j",
                "avg_power_w",
                "peak_power_w",
                "avg_duration_s",
                "avg_gross_energy_j",
                "avg_net_energy_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_model"],
                    row["mode"],
                    row["model_idx"],
                    row["run_count"],
                    f'{row["avg_batch_time_ms"]:.6f}',
                    f'{row["avg_gross_energy_per_batch_j"]:.6f}',
                    f'{row["avg_net_energy_per_batch_j"]:.6f}',
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["peak_power_w"]:.6f}',
                    f'{row["avg_duration_s"]:.6f}',
                    f'{row["avg_gross_energy_j"]:.6f}',
                    f'{row["avg_net_energy_j"]:.6f}',
                ]
            )


def make_summary(summary_rows, baseline_power):
    if not summary_rows:
        return "no summary rows"

    fastest = min(summary_rows, key=lambda row: row["avg_batch_time_ms"])
    lowest_net = min(summary_rows, key=lambda row: row["avg_net_energy_per_batch_j"])

    lines = [
        f'device_model = {summary_rows[0]["device_model"]}',
        f"baseline_power_w = {baseline_power:.6f}",
        "",
        "per_mode_model",
    ]

    for row in summary_rows:
        lines.append(
            "  mode={mode} model_idx={model_idx} avg_batch_time_ms={avg_batch_time_ms:.6f} "
            "avg_gross_energy_per_batch_j={avg_gross_energy_per_batch_j:.6f} "
            "avg_net_energy_per_batch_j={avg_net_energy_per_batch_j:.6f} "
            "avg_power_w={avg_power_w:.6f} peak_power_w={peak_power_w:.6f} "
            "avg_gross_energy_j={avg_gross_energy_j:.6f} avg_net_energy_j={avg_net_energy_j:.6f}".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "best",
            "  fastest = mode {mode} / model {model_idx} ({avg_batch_time_ms:.6f} ms/batch)".format(
                **fastest
            ),
            "  lowest_net_energy_per_batch = mode {mode} / model {model_idx} "
            "({avg_net_energy_per_batch_j:.6f} J/batch)".format(**lowest_net),
        ]
    )
    return "\n".join(lines)


def main():
    train_log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRAIN_LOG
    power_xlsx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_POWER_XLSX
    output_detail_csv, output_summary_csv, output_per_batch_csv, output_summary_txt = output_paths(
        train_log_path
    )

    samples = load_power_samples(power_xlsx_path)
    runs = load_runs(train_log_path)
    baseline_power = estimate_baseline(samples, runs)

    analyzed_runs = []
    for run in runs:
        result = analyze_run(run, samples, baseline_power)
        if result is not None:
            analyzed_runs.append(result)

    if not analyzed_runs:
        raise ValueError("no train runs matched any power samples")

    summary_rows = summarize_mode_model(analyzed_runs)
    write_detail_csv(analyzed_runs, output_detail_csv)
    write_summary_csv(summary_rows, output_summary_csv)
    write_per_batch_csv(summary_rows, output_per_batch_csv)

    summary = make_summary(summary_rows, baseline_power)
    output_summary_txt.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"detail csv written to: {output_detail_csv}")
    print(f"summary csv written to: {output_summary_csv}")
    print(f"per-batch csv written to: {output_per_batch_csv}")
    print(f"summary written to: {output_summary_txt}")


if __name__ == "__main__":
    main()
