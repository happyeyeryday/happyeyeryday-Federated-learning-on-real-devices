import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from models.SHFL_resnet import shfl_resnet18
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import Accumulator, accuracy
from utils.cama_real_profiles import (
    COOLDOWN_ROUNDS,
    FAILURE_TOLERANCE,
    LAG_TOLERANCE_SEC,
    cama_utility,
    choose_dvfs_label,
    choose_model_idx,
    device_strength,
    dvfs_mode_for,
    fairness_penalty,
    get_device_type,
    is_on_cooldown,
    model_depth_ratio_from_idx,
)
from utils.get_dataset import get_dataset
from utils.options import args_parser
from utils.power_manager_real import get_device_capacity, normalize_battery
from utils.set_seed import set_random_seed

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHECKPOINT_PATH = "checkpoint_cama_real.pth"


def now_str():
    return time.strftime(TIME_FORMAT)


def shfl_distribute(global_model_state, model_idx):
    local_state = {}
    target_exit_idx = model_idx - 1

    for k, v in global_model_state.items():
        if "num_batches_tracked" in k or "running_mean" in k or "running_var" in k:
            continue
        if k.startswith("conv1") or k.startswith("bn1"):
            local_state[k] = v.clone()
        elif k.startswith("mainblocks"):
            try:
                parts = k.split(".")
                block_id = int(parts[1])
                if block_id <= target_exit_idx:
                    local_state[k] = v.clone()
            except Exception:
                pass
        elif k.startswith("bottlenecks"):
            try:
                parts = k.split(".")
                exit_id = int(parts[1])
                if exit_id == target_exit_idx:
                    local_state["bottleneck." + ".".join(parts[2:])] = v.clone()
            except Exception:
                pass
        elif k.startswith("fcs"):
            try:
                parts = k.split(".")
                exit_id = int(parts[1])
                if exit_id == target_exit_idx:
                    local_state["fc." + ".".join(parts[2:])] = v.clone()
            except Exception:
                pass
    return local_state


def shfl_aggregate(w_local_list, model_indices, net_glob):
    global_state = net_glob.state_dict()
    sum_buffer = {}
    count_buffer = {}

    for k, v in global_state.items():
        if "num_batches_tracked" in k:
            continue
        sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
        count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    for idx, local_w in enumerate(w_local_list):
        model_idx = model_indices[idx]
        exit_idx = model_idx - 1
        for k_local, v_local in local_w.items():
            if k_local.startswith("bottleneck."):
                suffix = k_local[len("bottleneck."):]
                k_global = f"bottlenecks.{exit_idx}.{suffix}"
            elif k_local.startswith("fc."):
                suffix = k_local[len("fc."):]
                k_global = f"fcs.{exit_idx}.{suffix}"
            else:
                k_global = k_local

            if k_global not in sum_buffer:
                continue
            sum_buffer[k_global] += v_local
            count_buffer[k_global] += 1

    updated_state = {}
    for k in global_state.keys():
        if "num_batches_tracked" in k:
            updated_state[k] = global_state[k]
            continue
        updated_state[k] = global_state[k].clone()
        mask = count_buffer[k] > 0
        if mask.any():
            updated_state[k][mask] = (sum_buffer[k][mask] / count_buffer[k][mask]).to(global_state[k].dtype)
    return updated_state


def calibrate_bn(dataset, model, args):
    logger.info("Starting BN calibration...")
    calibrated = copy.deepcopy(model).to(args.device)
    calibrated.train()
    for module in calibrated.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.track_running_stats = True
            module.momentum = None

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            if idx >= 50:
                break
            calibrated(images.to(args.device))
    calibrated.eval()
    return calibrated


def summary_evaluate(net, dataset_test, device):
    net.eval()
    dt_loader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
    metrics = [Accumulator(2) for _ in range(4)]
    with torch.no_grad():
        for images, labels in dt_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            logits_list = outputs[0] if isinstance(outputs, tuple) else [outputs]
            for idx, pred in enumerate(logits_list):
                metrics[idx].add(accuracy(pred, labels), labels.numel())
    return [(metric[0] / metric[1]) * 100 for metric in metrics]


def parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except (TypeError, ValueError):
        return None


def duration_seconds(start_value, end_value):
    start_dt = parse_timestamp(start_value)
    end_dt = parse_timestamp(end_value)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def capacity_for_cid(cid):
    return get_device_capacity(get_device_type(cid))


def battery_ratio_for_cid(cid, joules):
    return normalize_battery(get_device_type(cid), joules)


def build_candidate_pool(active_clients, round_idx, target_m, failure_counter, last_selected_round):
    tolerated = [cid for cid in active_clients if failure_counter[cid] <= FAILURE_TOLERANCE]
    if not tolerated:
        return []

    ready = [
        cid
        for cid in tolerated
        if not is_on_cooldown(last_selected_round[cid], round_idx, COOLDOWN_ROUNDS)
    ]
    if len(ready) >= target_m:
        return ready
    return tolerated


def select_clients(
    candidate_ids,
    target_m,
    appearance_counter,
    weighted_participation,
    failure_counter,
    last_model_idx,
    last_round_duration,
):
    mean_weighted_participation = (
        float(np.mean([weighted_participation[cid] for cid in candidate_ids])) if candidate_ids else 0.0
    )
    scored = []

    for cid in candidate_ids:
        device_type = get_device_type(cid)
        lagged_last_round = (
            last_round_duration[cid] is not None and last_round_duration[cid] > LAG_TOLERANCE_SEC
        )
        model_idx = choose_model_idx(
            device_type=device_type,
            appearance_count=appearance_counter[cid],
            failure_count=failure_counter[cid],
            lagged_last_round=lagged_last_round,
            last_model_idx=last_model_idx[cid],
        )
        model_depth_ratio = model_depth_ratio_from_idx(model_idx)
        utility = cama_utility(
            device_type=device_type,
            model_idx=model_idx,
            weighted_participation=weighted_participation[cid],
            mean_weighted_participation=mean_weighted_participation,
        )
        if lagged_last_round:
            utility *= 0.85
        if failure_counter[cid] > 0:
            utility *= 0.9 ** failure_counter[cid]

        scored.append(
            {
                "cid": cid,
                "device_type": device_type,
                "model_idx": model_idx,
                "model_depth_ratio": model_depth_ratio,
                "utility": utility,
                "device_strength": device_strength(device_type),
                "fairness_penalty": fairness_penalty(
                    weighted_participation[cid], mean_weighted_participation
                ),
                "lagged_last_round": lagged_last_round,
                "weighted_participation": weighted_participation[cid],
            }
        )

    scored.sort(
        key=lambda item: (
            -item["utility"],
            item["weighted_participation"],
            -item["model_idx"],
            item["cid"],
        )
    )
    return scored[:target_m], mean_weighted_participation


def restore_battery_states(checkpoint_state, num_clients):
    restored = {}
    raw = checkpoint_state.get("battery_state_joules", {})
    for cid in range(num_clients):
        restored[cid] = min(max(float(raw.get(cid, capacity_for_cid(cid))), 0.0), capacity_for_cid(cid))
    return restored


def save_checkpoint(
    model,
    summary_records,
    best_acc,
    active_clients,
    retired_clients,
    battery_state_joules,
    appearance_counter,
    weighted_participation,
    last_selected_round,
    failure_counter,
    last_model_idx,
    last_dvfs_label,
    last_round_duration,
    last_upload_duration,
    last_status,
    round_idx,
):
    state = {
        "round": round_idx,
        "model_state_dict": model.state_dict(),
        "summary_records": summary_records,
        "best_acc": best_acc,
        "active_clients": active_clients,
        "retired_clients": sorted(retired_clients),
        "battery_state_joules": battery_state_joules,
        "battery_state_ratio": {
            cid: battery_ratio_for_cid(cid, battery_state_joules[cid]) for cid in battery_state_joules
        },
        "appearance_counter": appearance_counter,
        "weighted_participation": weighted_participation,
        "last_selected_round": last_selected_round,
        "failure_counter": failure_counter,
        "last_model_idx": last_model_idx,
        "last_dvfs_label": last_dvfs_label,
        "last_round_duration": last_round_duration,
        "last_upload_duration": last_upload_duration,
        "last_status": last_status,
    }
    torch.save(state, CHECKPOINT_PATH)


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)

    log_dir = Path("logs_real/cama_real_server")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"server_cama_real_{time.strftime('%Y%m%d_%H%M%S')}.log")

    num_clients = args.num_users
    target_m = max(int(args.frac * num_clients), 1)
    appearance_counter = {cid: 0 for cid in range(num_clients)}
    weighted_participation = {cid: 0.0 for cid in range(num_clients)}
    last_selected_round = {cid: None for cid in range(num_clients)}
    failure_counter = {cid: 0 for cid in range(num_clients)}
    last_model_idx = {cid: None for cid in range(num_clients)}
    last_dvfs_label = {cid: None for cid in range(num_clients)}
    last_round_duration = {cid: None for cid in range(num_clients)}
    last_upload_duration = {cid: None for cid in range(num_clients)}
    last_status = {cid: "idle" for cid in range(num_clients)}
    active_clients = list(range(num_clients))
    retired_clients = set()
    battery_state_joules = {cid: capacity_for_cid(cid) for cid in range(num_clients)}
    summary_records = []
    best_acc = 0.0
    start_round = 0

    logger.info("Starting CAMA real-device server")
    logger.info(
        f"device={args.device} num_clients={num_clients} frac={args.frac} "
        f"local_ep={args.local_ep} cooldown={COOLDOWN_ROUNDS} failure_tolerance={FAILURE_TOLERANCE}"
    )

    dataset_train, dataset_test, dict_users = get_dataset(args)
    net_glob = shfl_resnet18(num_classes=args.num_classes)
    net_glob.to(args.device)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=args.device)
        net_glob.load_state_dict(checkpoint["model_state_dict"], strict=False)
        summary_records = checkpoint.get("summary_records", [])
        best_acc = checkpoint.get("best_acc", 0.0)
        active_clients = checkpoint.get("active_clients", active_clients)
        retired_clients = set(checkpoint.get("retired_clients", []))
        battery_state_joules = restore_battery_states(checkpoint, num_clients)
        appearance_counter.update(checkpoint.get("appearance_counter", {}))
        weighted_participation.update(checkpoint.get("weighted_participation", {}))
        last_selected_round.update(checkpoint.get("last_selected_round", {}))
        failure_counter.update(checkpoint.get("failure_counter", {}))
        last_model_idx.update(checkpoint.get("last_model_idx", {}))
        last_dvfs_label.update(checkpoint.get("last_dvfs_label", {}))
        last_round_duration.update(checkpoint.get("last_round_duration", {}))
        last_upload_duration.update(checkpoint.get("last_upload_duration", {}))
        last_status.update(checkpoint.get("last_status", {}))
        start_round = int(checkpoint.get("round", -1)) + 1
        logger.info(f"Resumed from checkpoint {CHECKPOINT_PATH} at round {start_round}")

    connect_handler = ConnectHandler(num_clients, args.HOST, args.POST)

    for round_idx in range(start_round, args.epochs):
        round_start = time.time()
        candidate_ids = build_candidate_pool(
            active_clients=active_clients,
            round_idx=round_idx,
            target_m=target_m,
            failure_counter=failure_counter,
            last_selected_round=last_selected_round,
        )
        selected_plan, mean_weighted = select_clients(
            candidate_ids=candidate_ids,
            target_m=target_m,
            appearance_counter=appearance_counter,
            weighted_participation=weighted_participation,
            failure_counter=failure_counter,
            last_model_idx=last_model_idx,
            last_round_duration=last_round_duration,
        )
        if not selected_plan:
            logger.warning("No eligible CAMA candidates left, stopping.")
            break

        plan_by_cid = {}
        selected_ids = []
        send_failures = []
        for rank, item in enumerate(selected_plan):
            cid = item["cid"]
            item["dvfs_label"] = choose_dvfs_label(item["model_idx"], rank)
            item["dvfs_mode"] = dvfs_mode_for(item["device_type"], item["dvfs_label"])
            payload = {
                "type": "train_round",
                "round": round_idx,
                "net": shfl_distribute(net_glob.state_dict(), item["model_idx"]),
                "idxs_list": dict_users[cid],
                "model_idx": item["model_idx"],
                "model_depth_ratio": item["model_depth_ratio"],
                "dvfs_label": item["dvfs_label"],
                "dvfs_mode": item["dvfs_mode"],
                "local_ep": args.local_ep,
                "lr": args.lr * (args.lr_decay ** round_idx),
                "battery_joules": battery_state_joules[cid],
                "battery_level": battery_ratio_for_cid(cid, battery_state_joules[cid]),
            }
            logger.info(
                f"{now_str()} ROUND={round_idx} SEND cid={cid} type={item['device_type']} "
                f"model_idx={item['model_idx']} depth_ratio={item['model_depth_ratio']:.2f} "
                f"dvfs_mode={item['dvfs_mode']} utility={item['utility']:.4f}"
            )
            if connect_handler.sendData(cid, payload):
                selected_ids.append(cid)
                plan_by_cid[cid] = item
                last_selected_round[cid] = round_idx
                last_model_idx[cid] = item["model_idx"]
                last_dvfs_label[cid] = item["dvfs_label"]
            else:
                failure_counter[cid] += 1
                last_status[cid] = "send_failed"
                send_failures.append({"cid": cid, "reason": "send_failed"})

        if not selected_ids:
            logger.warning("No selected CAMA clients received the round payload, stopping.")
            break

        logger.info(
            f"{now_str()} ROUND={round_idx} candidates={candidate_ids} selected={selected_ids} "
            f"types={[plan_by_cid[c]['device_type'] for c in selected_ids]} "
            f"model_indices={[plan_by_cid[c]['model_idx'] for c in selected_ids]} "
            f"depth_ratios={[plan_by_cid[c]['model_depth_ratio'] for c in selected_ids]} "
            f"dvfs_modes={[plan_by_cid[c]['dvfs_mode'] for c in selected_ids]} "
            f"mean_weighted={mean_weighted:.4f}"
        )

        local_models = []
        local_indices = []
        responses = 0
        expected = len(selected_ids)
        upload_order = []
        successful_clients = []
        round_failures = list(send_failures)
        round_retired = []

        while responses < expected:
            msg, cid = connect_handler.receiveData()
            msg_type = msg.get("type")

            if "battery_joules" in msg:
                battery_state_joules[cid] = min(
                    max(float(msg["battery_joules"]), 0.0),
                    capacity_for_cid(cid),
                )
            elif "battery_level" in msg:
                battery_state_joules[cid] = min(
                    max(float(msg["battery_level"]), 0.0),
                    1.0,
                ) * capacity_for_cid(cid)

            if msg_type == "client_update" and msg.get("round") == round_idx and cid in plan_by_cid:
                responses += 1
                upload_order.append(cid)
                successful_clients.append(cid)
                local_models.append(msg["net"])
                local_indices.append(int(msg["model_idx"]))

                train_duration = duration_seconds(msg.get("train_start_time"), msg.get("train_end_time"))
                upload_duration = duration_seconds(msg.get("upload_start_time"), msg.get("upload_end_time"))
                last_round_duration[cid] = train_duration
                last_upload_duration[cid] = upload_duration
                appearance_counter[cid] += 1
                weighted_participation[cid] += float(msg.get("model_depth_ratio", int(msg["model_idx"]) / 4.0))
                failure_counter[cid] = 0
                last_status[cid] = msg.get("status", "ok")

                logger.info(
                    f"{now_str()} ROUND={round_idx} RECV cid={cid} status={msg.get('status', 'ok')} "
                    f"train=({msg.get('train_start_time')},{msg.get('train_end_time')}) "
                    f"upload=({msg.get('upload_start_time')},{msg.get('upload_end_time')}) "
                    f"dvfs_mode={msg.get('dvfs_mode')} battery={battery_state_joules[cid]:.2f}J"
                )
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx})
            elif msg_type == "status" and msg.get("status") == "low_battery" and cid in plan_by_cid:
                responses += 1
                retired_clients.add(cid)
                if cid in active_clients:
                    active_clients.remove(cid)
                round_retired.append(cid)
                last_status[cid] = "low_battery"
                connect_handler.sendData(cid, {"type": "shutdown_ack", "round": round_idx})
                logger.warning(
                    f"Client {cid} low battery at round {round_idx}, retiring with "
                    f"{battery_state_joules[cid]:.2f}J"
                )
            elif msg_type == "client_error" and msg.get("round") == round_idx and cid in plan_by_cid:
                responses += 1
                failure_counter[cid] += 1
                last_status[cid] = msg.get("status", "client_error")
                round_failures.append(
                    {"cid": cid, "reason": msg.get("failure_reason", msg.get("reason", "client_error"))}
                )
                logger.warning(
                    f"Client {cid} failed round {round_idx}: "
                    f"{msg.get('failure_reason', msg.get('reason', 'client_error'))}"
                )
            else:
                logger.warning(f"Unknown message from cid={cid}: {msg}")

        if local_models:
            net_glob.load_state_dict(shfl_aggregate(local_models, local_indices, net_glob), strict=False)

        calibrated_model = calibrate_bn(dataset_train, net_glob, args)
        acc_list = summary_evaluate(calibrated_model, dataset_test, args.device)
        net_glob = calibrated_model
        best_acc = max(best_acc, acc_list[-1])
        round_duration = time.time() - round_start

        record = {
            "round": round_idx,
            "candidate_clients": candidate_ids,
            "selected_clients": selected_ids,
            "selected_device_types": [plan_by_cid[c]["device_type"] for c in selected_ids],
            "selected_model_indices": [plan_by_cid[c]["model_idx"] for c in selected_ids],
            "selected_model_depth_ratios": [plan_by_cid[c]["model_depth_ratio"] for c in selected_ids],
            "selected_dvfs_modes": [plan_by_cid[c]["dvfs_mode"] for c in selected_ids],
            "successful_clients": successful_clients,
            "failed_clients": round_failures,
            "retired_clients": round_retired,
            "upload_order": upload_order,
            "mean_weighted_participation": mean_weighted,
            "battery_state_joules": {cid: battery_state_joules[cid] for cid in range(num_clients)},
            "acc_all_models": acc_list,
            "acc_model4": acc_list[-1],
            "best_acc": best_acc,
            "round_duration_sec": round_duration,
        }
        summary_records.append(record)
        logger.info(
            f"{now_str()} ROUND={round_idx} acc_model4={acc_list[-1]:.4f} "
            f"all_acc={acc_list} best_acc={best_acc:.4f} duration={round_duration:.2f}s "
            f"active={active_clients}"
        )
        save_checkpoint(
            model=net_glob,
            summary_records=summary_records,
            best_acc=best_acc,
            active_clients=active_clients,
            retired_clients=retired_clients,
            battery_state_joules=battery_state_joules,
            appearance_counter=appearance_counter,
            weighted_participation=weighted_participation,
            last_selected_round=last_selected_round,
            failure_counter=failure_counter,
            last_model_idx=last_model_idx,
            last_dvfs_label=last_dvfs_label,
            last_round_duration=last_round_duration,
            last_upload_duration=last_upload_duration,
            last_status=last_status,
            round_idx=round_idx,
        )

    for cid in range(num_clients):
        try:
            connect_handler.sendData(cid, {"type": "stop"})
        except Exception:
            pass

    summary_path = log_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "algorithm": "cama_real",
        "model_family": "shfl_resnet18",
        "heterogeneity": "depth/model_idx",
        "training_objective": "single_exit_cross_entropy",
        "self_distillation": False,
        "dvfs_policy": "helcfl_aligned_fixed_rule",
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "frac": args.frac,
        "local_ep": args.local_ep,
        "records": summary_records,
        "best_acc": best_acc,
        "final_acc": summary_records[-1]["acc_model4"] if summary_records else 0.0,
        "checkpoint_path": CHECKPOINT_PATH,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved summary to {summary_path}")
