#!/usr/bin/env python3
"""
静息功耗分析脚本。

默认输入：
1. scripts/nano/idle/nanoidle.csv
2. scripts/nano/idle/nano_idle.xlsx

支持命令行覆盖：
python3 scripts/analyze_idle_power.py <idle_log.csv> <power.xlsx>
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree as ET
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "nano" / "idle"
DEFAULT_IDLE_LOG = DEFAULT_INPUT_DIR / "nanoidle.csv"
DEFAULT_POWER_XLSX = DEFAULT_INPUT_DIR / "nano_idle.xlsx"

TIME_COLUMN = "A"
POWER_COLUMN = "E"
EXCEL_EPOCH = datetime(1899, 12, 30)

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def output_paths(idle_log_path):
    out_dir = idle_log_path.resolve().parent
    return (
        out_dir / "idle_power_detail.csv",
        out_dir / "idle_power_result.csv",
        out_dir / "idle_power_summary.txt",
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


def normalize_text(text):
    return text.replace("\x00", "").replace("\r", "").strip()


def parse_idle_line(line, line_number):
    text = normalize_text(line)
    if not text:
        return None
    if text.startswith("device_model,"):
        return None
    if text.startswith('"') and text.count('"') % 2 == 1:
        text = text[1:]

    parts = text.rsplit(",", 8)
    if len(parts) != 9:
        raise ValueError(f"invalid idle log row at line {line_number}: {text}")

    return {
        "device_model": parts[0],
        "mode": int(parts[1]),
        "stabilize_seconds": int(parts[2]),
        "measure_seconds": int(parts[3]),
        "start": datetime.strptime(parts[4], "%Y-%m-%d %H:%M:%S"),
        "end": datetime.strptime(parts[5], "%Y-%m-%d %H:%M:%S"),
        "duration_s": float(parts[6]),
        "status": parts[7],
        "error": parts[8],
    }


def load_idle_windows(idle_log_path):
    windows = []
    with idle_log_path.open("r", encoding="utf-8", newline="") as fh:
        for line_number, line in enumerate(fh, start=1):
            row = parse_idle_line(line, line_number)
            if row is None:
                continue
            if row["status"] != "ok":
                continue
            windows.append(row)

    if not windows:
        raise ValueError(f"no successful idle windows found in {idle_log_path}")
    return windows


def analyze_window(window, samples):
    matched = [sample for sample in samples if window["start"] <= sample["time"] <= window["end"]]
    if not matched:
        return None

    avg_power = mean(sample["power_w"] for sample in matched)
    min_power = min(sample["power_w"] for sample in matched)
    max_power = max(sample["power_w"] for sample in matched)
    energy_j = avg_power * window["duration_s"]

    result = dict(window)
    result["sample_count"] = len(matched)
    result["avg_power_w"] = avg_power
    result["min_power_w"] = min_power
    result["max_power_w"] = max_power
    result["energy_j"] = energy_j
    return result


def summarize_modes(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    summaries = []
    for mode in sorted(grouped):
        items = grouped[mode]
        summaries.append(
            {
                "device_model": items[0]["device_model"],
                "mode": mode,
                "run_count": len(items),
                "avg_duration_s": mean(item["duration_s"] for item in items),
                "avg_power_w": mean(item["avg_power_w"] for item in items),
                "min_power_w": min(item["min_power_w"] for item in items),
                "max_power_w": max(item["max_power_w"] for item in items),
                "avg_energy_j": mean(item["energy_j"] for item in items),
                "avg_sample_count": mean(item["sample_count"] for item in items),
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
                "stabilize_seconds",
                "measure_seconds",
                "start_time",
                "end_time",
                "duration_s",
                "sample_count",
                "avg_power_w",
                "min_power_w",
                "max_power_w",
                "energy_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_model"],
                    row["mode"],
                    row["stabilize_seconds"],
                    row["measure_seconds"],
                    row["start"].strftime("%Y-%m-%d %H:%M:%S"),
                    row["end"].strftime("%Y-%m-%d %H:%M:%S"),
                    f'{row["duration_s"]:.6f}',
                    row["sample_count"],
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["min_power_w"]:.6f}',
                    f'{row["max_power_w"]:.6f}',
                    f'{row["energy_j"]:.6f}',
                ]
            )


def write_result_csv(rows, output_result_csv):
    with output_result_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "device_model",
                "mode",
                "run_count",
                "avg_duration_s",
                "avg_sample_count",
                "avg_power_w",
                "min_power_w",
                "max_power_w",
                "avg_energy_j",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_model"],
                    row["mode"],
                    row["run_count"],
                    f'{row["avg_duration_s"]:.6f}',
                    f'{row["avg_sample_count"]:.6f}',
                    f'{row["avg_power_w"]:.6f}',
                    f'{row["min_power_w"]:.6f}',
                    f'{row["max_power_w"]:.6f}',
                    f'{row["avg_energy_j"]:.6f}',
                ]
            )


def make_summary(rows):
    if not rows:
        return "no mode rows"

    lowest_power = min(rows, key=lambda row: row["avg_power_w"])
    device_model = rows[0]["device_model"]

    lines = [
        f"device_model = {device_model}",
        "",
        "per_mode",
    ]
    for row in rows:
        lines.append(
            "  mode={mode} run_count={run_count} avg_duration_s={avg_duration_s:.6f} "
            "avg_sample_count={avg_sample_count:.6f} avg_power_w={avg_power_w:.6f} "
            "min_power_w={min_power_w:.6f} max_power_w={max_power_w:.6f} "
            "avg_energy_j={avg_energy_j:.6f}".format(**row)
        )

    lines.extend(
        [
            "",
            "best",
            f"  lowest_idle_power_mode = {lowest_power['mode']} ({lowest_power['avg_power_w']:.6f} W)",
        ]
    )
    return "\n".join(lines)


def main():
    idle_log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IDLE_LOG
    power_xlsx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_POWER_XLSX
    output_detail_csv, output_result_csv, output_summary = output_paths(idle_log_path)

    samples = load_power_samples(power_xlsx_path)
    windows = load_idle_windows(idle_log_path)

    analyzed_rows = []
    for window in windows:
        result = analyze_window(window, samples)
        if result is not None:
            analyzed_rows.append(result)

    if not analyzed_rows:
        raise ValueError("no idle windows matched any power samples")

    summary_rows = summarize_modes(analyzed_rows)
    write_detail_csv(analyzed_rows, output_detail_csv)
    write_result_csv(summary_rows, output_result_csv)

    summary = make_summary(summary_rows)
    output_summary.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"detail csv written to: {output_detail_csv}")
    print(f"result csv written to: {output_result_csv}")
    print(f"summary written to: {output_summary}")


if __name__ == "__main__":
    main()
