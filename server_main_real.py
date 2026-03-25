import copy
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from models.SHFL_resnet import shfl_resnet18
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import Accumulator, accuracy
from utils.get_dataset import get_dataset
from utils.helcfl_real_profiles import get_device_type
from utils.main_real_policy import MainRealPolicy, estimate_idle_drain_joules
from utils.options import args_parser
from utils.power_manager_real import LOW_BATTERY_THRESHOLD_J, get_device_capacity, normalize_battery
from utils.set_seed import set_random_seed

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHECKPOINT_PATH = "checkpoint_main_real.pth"


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
                block_id = int(k.split(".")[1])
            except Exception:
                continue
            if block_id <= target_exit_idx:
                local_state[k] = v.clone()
        elif k.startswith("bottlenecks"):
            try:
                exit_id = int(k.split(".")[1])
            except Exception:
                continue
            if exit_id == target_exit_idx:
                local_state["bottleneck." + ".".join(k.split(".")[2:])] = v.clone()
        elif k.startswith("fcs"):
            try:
                exit_id = int(k.split(".")[1])
            except Exception:
                continue
            if exit_id == target_exit_idx:
                local_state["fc." + ".".join(k.split(".")[2:])] = v.clone()
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

    for local_w, model_idx in zip(w_local_list, model_indices):
        exit_idx = int(model_idx) - 1
        for k_local, v_local in local_w.items():
            if k_local.startswith("bottleneck."):
                suffix = k_local[len("bottleneck.") :]
                k_global = f"bottlenecks.{exit_idx}.{suffix}"
            elif k_local.startswith("fc."):
                suffix = k_local[len("fc.") :]
                k_global = f"fcs.{exit_idx}.{suffix}"
            else:
                k_global = k_local
            if k_global not in sum_buffer:
                continue
            sum_buffer[k_global] += v_local
            count_buffer[k_global] += 1

    updated_state = {}
    for k, v in global_state.items():
        if "num_batches_tracked" in k:
            updated_state[k] = v
            continue
        updated_state[k] = v.clone()
        mask = count_buffer[k] > 0
        if mask.any():
            updated_state[k][mask] = (sum_buffer[k][mask] / count_buffer[k][mask]).to(v.dtype)
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


def capacity_for_cid(cid):
    return get_device_capacity(get_device_type(cid))


def battery_ratio_for_cid(cid, joules):
    return normalize_battery(get_device_type(cid), joules)


def restore_battery_states(checkpoint_state, num_clients):
    restored = {}
    raw = checkpoint_state.get("battery_state_joules", {})
    for cid in range(num_clients):
        restored[cid] = min(max(float(raw.get(cid, capacity_for_cid(cid))), 0.0), capacity_for_cid(cid))
    return restored


def save_checkpoint(
    model,
    policy,
    summary_records,
    best_acc,
    active_clients,
    retired_clients,
    battery_state_joules,
    last_dvfs_labels,
    round_idx,
):
    state = {
        "round": round_idx,
        "model_state_dict": model.state_dict(),
        "policy_state": policy.get_state(),
        "summary_records": summary_records,
        "best_acc": best_acc,
        "active_clients": list(active_clients),
        "retired_clients": sorted(retired_clients),
        "battery_state_joules": battery_state_joules,
        "battery_state_ratio": {
            cid: battery_ratio_for_cid(cid, battery_state_joules[cid]) for cid in battery_state_joules
        },
        "last_dvfs_labels": dict(last_dvfs_labels),
    }
    torch.save(state, CHECKPOINT_PATH)


if __name__ == "__main__":
    args = args_parser()
    if args.policy_mode != "offline_bundle":
        raise ValueError(f"unsupported policy_mode={args.policy_mode}")
    if not args.policy_bundle:
        raise ValueError("--policy_bundle is required for server_main_real.py")

    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)

    log_dir = Path("logs_real/main_real_server")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"server_main_real_{time.strftime('%Y%m%d_%H%M%S')}.log")

    num_clients = args.num_users
    active_clients = list(range(num_clients))
    retired_clients = set()
    battery_state_joules = {cid: capacity_for_cid(cid) for cid in range(num_clients)}
    last_dvfs_labels = {cid: "low" for cid in range(num_clients)}
    summary_records = []
    best_acc = 0.0
    start_round = 0

    logger.info("Starting main real-device server")
    logger.info(
        f"device={args.device} num_clients={num_clients} local_ep={args.local_ep} "
        f"policy_bundle={args.policy_bundle}"
    )

    dataset_train, dataset_test, dict_users = get_dataset(args)
    net_glob = shfl_resnet18(num_classes=args.num_classes)
    net_glob.to(args.device)

    policy = MainRealPolicy(
        bundle_dir=args.policy_bundle,
        manifest_path=args.policy_manifest or None,
        device=args.device,
        num_clients=num_clients,
    )

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=args.device)
        net_glob.load_state_dict(checkpoint["model_state_dict"], strict=False)
        summary_records = checkpoint.get("summary_records", [])
        best_acc = float(checkpoint.get("best_acc", 0.0))
        active_clients = list(checkpoint.get("active_clients", active_clients))
        retired_clients = set(checkpoint.get("retired_clients", []))
        battery_state_joules = restore_battery_states(checkpoint, num_clients)
        last_dvfs_labels.update(checkpoint.get("last_dvfs_labels", {}))
        if "policy_state" in checkpoint:
            policy.load_state(checkpoint["policy_state"])
        start_round = int(checkpoint.get("round", -1)) + 1
        logger.info(f"Resumed from checkpoint {CHECKPOINT_PATH} at round {start_round}")

    connect_handler = ConnectHandler(num_clients, args.HOST, args.POST)

    for round_idx in range(start_round, args.epochs):
        round_start = time.time()
        active_clients = [cid for cid in active_clients if battery_state_joules[cid] > LOW_BATTERY_THRESHOLD_J]
        if not active_clients:
            logger.warning("No active clients left, stopping.")
            break

        plan_by_cid = policy.plan_round(active_clients=active_clients, battery_state_joules=battery_state_joules)
        planned_selected_ids = [cid for cid in active_clients if plan_by_cid[cid]["selected"]]
        idle_ids = [cid for cid in active_clients if not plan_by_cid[cid]["selected"]]
        if not planned_selected_ids:
            logger.warning("Policy selected no trainable clients, stopping.")
            break

        logger.info(
            f"{now_str()} ROUND={round_idx} planned_selected={planned_selected_ids} idle={idle_ids} "
            f"model_indices={[plan_by_cid[c]['model_idx'] for c in planned_selected_ids]} "
            f"dvfs_modes={[plan_by_cid[c]['dvfs_mode'] for c in planned_selected_ids]} "
            f"roles={[plan_by_cid[c]['role_id'] for c in active_clients]}"
        )

        selected_ids = []
        for cid in planned_selected_ids:
            item = plan_by_cid[cid]
            payload = {
                "type": "train_round",
                "round": round_idx,
                "net": shfl_distribute(net_glob.state_dict(), item["model_idx"]),
                "idxs_list": dict_users[cid],
                "model_idx": item["model_idx"],
                "model_depth_ratio": item["model_depth_ratio"],
                "dvfs_mode": item["dvfs_mode"],
                "local_ep": args.local_ep,
                "lr": args.lr * (args.lr_decay ** round_idx),
                "battery_joules": battery_state_joules[cid],
                "battery_level": battery_ratio_for_cid(cid, battery_state_joules[cid]),
            }
            if not connect_handler.sendData(cid, payload):
                logger.warning(f"Failed to send round {round_idx} to cid={cid}, skipping.")
                idle_ids.append(cid)
                plan_by_cid[cid]["selected"] = False
                continue
            selected_ids.append(cid)

        if not selected_ids:
            logger.warning("No selected clients received the round payload, stopping.")
            break

        local_models = []
        local_indices = []
        upload_order = []
        round_retired = []
        responses = 0
        expected = len(selected_ids)
        while responses < expected:
            msg, cid = connect_handler.receiveData()
            msg_type = msg.get("type")

            if "battery_joules" in msg:
                battery_state_joules[cid] = min(max(float(msg["battery_joules"]), 0.0), capacity_for_cid(cid))
            elif "battery_level" in msg:
                battery_state_joules[cid] = min(max(float(msg["battery_level"]), 0.0), 1.0) * capacity_for_cid(cid)

            if msg_type == "client_update" and msg.get("round") == round_idx:
                responses += 1
                upload_order.append(cid)
                local_models.append(msg["net"])
                local_indices.append(int(msg["model_idx"]))
                last_dvfs_labels[cid] = str(plan_by_cid[cid]["dvfs_label"])
                logger.info(
                    f"{now_str()} RECV cid={cid} train=({msg.get('train_start_time')},{msg.get('train_end_time')}) "
                    f"upload=({msg.get('upload_start_time')},{msg.get('upload_end_time')}) "
                    f"dvfs_mode={msg.get('dvfs_mode')} battery={battery_state_joules[cid]:.2f}J"
                )
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx})
            elif msg_type == "status" and msg.get("status") == "low_battery":
                responses += 1
                retired_clients.add(cid)
                if cid in active_clients:
                    active_clients.remove(cid)
                round_retired.append(cid)
                logger.warning(f"Client {cid} low battery at round {round_idx}, retiring.")
                connect_handler.sendData(cid, {"type": "shutdown_ack", "round": round_idx})
            elif msg_type == "client_error":
                responses += 1
                logger.warning(f"Client {cid} failed round {round_idx}: {msg.get('reason')}")
            else:
                logger.warning(f"Unknown message from cid={cid}: {msg}")

        round_duration = time.time() - round_start
        for cid in idle_ids:
            if cid in retired_clients:
                continue
            drained = estimate_idle_drain_joules(get_device_type(cid), last_dvfs_labels.get(cid, "low"), round_duration)
            battery_state_joules[cid] = max(0.0, battery_state_joules[cid] - drained)
            if battery_state_joules[cid] <= LOW_BATTERY_THRESHOLD_J:
                retired_clients.add(cid)
                if cid in active_clients:
                    active_clients.remove(cid)
                round_retired.append(cid)

        if local_models:
            net_glob.load_state_dict(shfl_aggregate(local_models, local_indices, net_glob), strict=False)

        calibrated_model = calibrate_bn(dataset_train, net_glob, args)
        acc_list = summary_evaluate(calibrated_model, dataset_test, args.device)
        net_glob = calibrated_model
        best_acc = max(best_acc, float(acc_list[-1]))

        record = {
            "round": round_idx,
            "selected_clients": selected_ids,
            "idle_clients": idle_ids,
            "selected_device_types": [plan_by_cid[c]["device_type"] for c in selected_ids],
            "selected_model_indices": [plan_by_cid[c]["model_idx"] for c in selected_ids],
            "selected_model_depth_ratios": [plan_by_cid[c]["model_depth_ratio"] for c in selected_ids],
            "selected_dvfs_modes": [plan_by_cid[c]["dvfs_mode"] for c in selected_ids],
            "selected_dvfs_labels": [plan_by_cid[c]["dvfs_label"] for c in selected_ids],
            "role_ids": {cid: int(plan_by_cid[cid]["role_id"]) for cid in active_clients},
            "action_ids": {cid: int(plan_by_cid[cid]["action_id"]) for cid in active_clients},
            "upload_order": upload_order,
            "retired_clients": round_retired,
            "battery_state_joules": {cid: battery_state_joules[cid] for cid in range(num_clients)},
            "acc_all_models": acc_list,
            "acc_model4": acc_list[-1],
            "best_acc": best_acc,
            "round_duration_sec": round_duration,
        }
        summary_records.append(record)
        logger.info(
            f"{now_str()} ROUND={round_idx} acc_model4={acc_list[-1]:.4f} best_acc={best_acc:.4f} "
            f"duration={round_duration:.2f}s active={sorted(active_clients)}"
        )
        save_checkpoint(
            model=net_glob,
            policy=policy,
            summary_records=summary_records,
            best_acc=best_acc,
            active_clients=active_clients,
            retired_clients=retired_clients,
            battery_state_joules=battery_state_joules,
            last_dvfs_labels=last_dvfs_labels,
            round_idx=round_idx,
        )

    for cid in range(num_clients):
        try:
            connect_handler.sendData(cid, {"type": "stop"})
        except Exception:
            pass

    summary_path = log_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "algorithm": "main_real",
        "policy_mode": args.policy_mode,
        "policy_bundle": args.policy_bundle,
        "policy_manifest": args.policy_manifest or str(Path(args.policy_bundle) / "policy_manifest.json"),
        "model_family": "shfl_resnet18",
        "heterogeneity": "depth/model_idx",
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "local_ep": args.local_ep,
        "records": summary_records,
        "best_acc": best_acc,
        "final_acc": summary_records[-1]["acc_model4"] if summary_records else 0.0,
        "checkpoint_path": CHECKPOINT_PATH,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved summary to {summary_path}")
