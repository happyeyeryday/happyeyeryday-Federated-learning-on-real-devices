import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from models.hetero_model import resnet18
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import Accumulator, accuracy
from utils.get_dataset import DatasetSplit, get_dataset
from utils.helcfl_real_profiles import (
    choose_dvfs_label,
    choose_model_id,
    dvfs_mode_for,
    estimated_cost,
    get_device_type,
    model_rate_from_id,
)
from utils.options import args_parser
from utils.set_seed import set_random_seed


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)


def get_split_indices(global_k, global_v, rate):
    if "num_batches_tracked" in global_k:
        return None, None

    dims = global_v.dim()
    full_out = global_v.size(0)
    is_classifier = ("linear" in global_k) or ("fc" in global_k) or ("classifier" in global_k)
    split_out = full_out if is_classifier else int(np.ceil(full_out * rate))

    if dims == 1:
        return slice(0, split_out), None

    full_in = global_v.size(1)
    split_in = int(np.ceil(full_in * rate))
    if dims == 4 and full_in == 3:
        split_in = full_in
    return slice(0, split_out), slice(0, split_in)


def hetero_distribute(global_model_state, rate):
    local_state = {}
    for k, v in global_model_state.items():
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            continue
        slice_0, slice_1 = get_split_indices(k, v, rate)
        if slice_0 is None:
            continue
        if v.dim() == 1:
            local_state[k] = v[slice_0].clone()
        elif v.dim() == 2:
            local_state[k] = v[slice_0, slice_1].clone()
        elif v.dim() == 4:
            local_state[k] = v[slice_0, slice_1, :, :].clone()
    return local_state


def hetero_aggregate(w_local_list, rates_list, net_glob):
    global_state = net_glob.state_dict()
    sum_buffer = {}
    count_buffer = {}

    for k, v in global_state.items():
        if "num_batches_tracked" in k:
            sum_buffer[k] = v.clone()
            count_buffer[k] = torch.ones_like(v)
        else:
            sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
            count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    for idx, local_w in enumerate(w_local_list):
        rate = rates_list[idx]
        for k, v_local in local_w.items():
            if k not in sum_buffer or "num_batches_tracked" in k:
                continue
            global_v = global_state[k]
            slice_0, slice_1 = get_split_indices(k, global_v, rate)
            if global_v.dim() == 1:
                sum_buffer[k][slice_0] += v_local
                count_buffer[k][slice_0] += 1
            elif global_v.dim() == 2:
                sum_buffer[k][slice_0, slice_1] += v_local
                count_buffer[k][slice_0, slice_1] += 1
            elif global_v.dim() == 4:
                sum_buffer[k][slice_0, slice_1, :, :] += v_local
                count_buffer[k][slice_0, slice_1, :, :] += 1

    updated_state = {}
    for k in global_state.keys():
        if "num_batches_tracked" in k:
            updated_state[k] = sum_buffer[k]
            continue
        updated_state[k] = global_state[k].clone()
        mask_updated = count_buffer[k] > 0
        if mask_updated.any():
            updated_state[k][mask_updated] = (
                sum_buffer[k][mask_updated] / count_buffer[k][mask_updated]
            ).to(global_state[k].dtype)
    return updated_state


def calibrate_bn(dataset, model, args):
    calibrated = copy.deepcopy(model).to(args.device)
    for module in calibrated.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.track_running_stats:
                module.reset_running_stats()
            module.track_running_stats = True
            module.training = True

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            if idx >= 50:
                break
            calibrated(images.to(args.device))
    calibrated.eval()
    return calibrated


def evaluate(net, dataset_test, device):
    net.eval()
    dt_loader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
    metric = Accumulator(2)
    with torch.no_grad():
        for images, labels in dt_loader:
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            metric.add(accuracy(logits, labels), labels.numel())
    return metric[0] / metric[1]


def select_clients(active_clients, target_m, appearance_counter, utility_eta):
    scored = []
    for cid in active_clients:
        device_type = get_device_type(cid)
        model_id = choose_model_id(device_type, appearance_counter[cid])
        utility = (utility_eta ** appearance_counter[cid]) / estimated_cost(device_type, model_id)
        scored.append((utility, cid, model_id, device_type))
    scored.sort(reverse=True)
    return scored[:target_m]


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)

    log_dir = Path("logs_real/helcfl_real_server")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"server_helcfl_real_{time.strftime('%Y%m%d_%H%M%S')}.log")

    utility_eta = 0.9
    num_clients = args.num_users
    target_m = max(int(args.frac * num_clients), 1)
    appearance_counter = {cid: 0 for cid in range(num_clients)}
    active_clients = list(range(num_clients))
    summary_records = []
    best_acc = 0.0

    logger.info("Starting HELCFL real-device server")
    logger.info(f"device={args.device} num_clients={num_clients} frac={args.frac} local_ep={args.local_ep}")

    dataset_train, dataset_test, dict_users = get_dataset(args)
    net_glob = resnet18(model_rate=1.0, track=True)
    net_glob.apply(init_weights)
    net_glob.to(args.device)

    connect_handler = ConnectHandler(num_clients, args.HOST, args.POST)

    for round_idx in range(args.epochs):
        round_start = time.time()
        selected_plan = select_clients(active_clients, target_m, appearance_counter, utility_eta)
        selected_ids = [cid for _, cid, _, _ in selected_plan]
        if not selected_ids:
            logger.warning("No active clients left, stopping.")
            break

        plan_by_cid = {}
        for rank, (_, cid, model_id, device_type) in enumerate(selected_plan):
            model_rate = model_rate_from_id(model_id)
            dvfs_label = choose_dvfs_label(model_id, rank)
            dvfs_mode = dvfs_mode_for(device_type, dvfs_label)
            appearance_counter[cid] += 1
            plan_by_cid[cid] = {
                "device_type": device_type,
                "model_id": model_id,
                "model_rate": model_rate,
                "dvfs_label": dvfs_label,
                "dvfs_mode": dvfs_mode,
            }

        logger.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ROUND={round_idx} "
            f"selected={selected_ids} "
            f"model_ids={[plan_by_cid[c]['model_id'] for c in selected_ids]} "
            f"model_rates={[plan_by_cid[c]['model_rate'] for c in selected_ids]} "
            f"dvfs_modes={[plan_by_cid[c]['dvfs_mode'] for c in selected_ids]}"
        )

        for cid in selected_ids:
            item = plan_by_cid[cid]
            payload = {
                "type": "train_round",
                "round": round_idx,
                "net": hetero_distribute(net_glob.state_dict(), item["model_rate"]),
                "idxs_list": dict_users[cid],
                "model_id": item["model_id"],
                "model_rate": item["model_rate"],
                "dvfs_label": item["dvfs_label"],
                "dvfs_mode": item["dvfs_mode"],
                "local_ep": args.local_ep,
                "lr": args.lr * (args.lr_decay ** round_idx),
            }
            logger.info(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} SEND cid={cid} type={item['device_type']} "
                f"model_id={item['model_id']} rate={item['model_rate']} dvfs_mode={item['dvfs_mode']}"
            )
            connect_handler.sendData(cid, payload)

        local_models = []
        local_rates = []
        responses = 0
        expected = len(selected_ids)
        upload_order = []
        while responses < expected:
            msg, cid = connect_handler.receiveData()
            if msg.get("type") == "client_update" and msg.get("round") == round_idx:
                responses += 1
                upload_order.append(cid)
                local_models.append(msg["net"])
                local_rates.append(float(msg["model_rate"]))
                logger.info(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} RECV cid={cid} "
                    f"train=({msg.get('train_start_time')},{msg.get('train_end_time')}) "
                    f"upload=({msg.get('upload_start_time')},{msg.get('upload_end_time')}) "
                    f"dvfs_mode={msg.get('dvfs_mode')}"
                )
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx})
            elif msg.get("type") == "client_error":
                responses += 1
                logger.warning(f"Client {cid} failed round {round_idx}: {msg.get('reason')}")
            else:
                logger.warning(f"Unknown message from cid={cid}: {msg}")

        if local_models:
            net_glob.load_state_dict(hetero_aggregate(local_models, local_rates, net_glob), strict=False)

        calibrated_model = calibrate_bn(dataset_train, net_glob, args)
        acc = evaluate(calibrated_model, dataset_test, args.device) * 100
        net_glob = calibrated_model
        best_acc = max(best_acc, acc)
        round_duration = time.time() - round_start

        record = {
            "round": round_idx,
            "selected_clients": selected_ids,
            "selected_device_types": [plan_by_cid[c]["device_type"] for c in selected_ids],
            "selected_model_ids": [plan_by_cid[c]["model_id"] for c in selected_ids],
            "selected_model_rates": [plan_by_cid[c]["model_rate"] for c in selected_ids],
            "selected_dvfs_modes": [plan_by_cid[c]["dvfs_mode"] for c in selected_ids],
            "upload_order": upload_order,
            "acc": acc,
            "best_acc": best_acc,
            "round_duration_sec": round_duration,
        }
        summary_records.append(record)
        logger.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ROUND={round_idx} "
            f"acc={acc:.4f} best_acc={best_acc:.4f} duration={round_duration:.2f}s"
        )

    for cid in range(num_clients):
        try:
            connect_handler.sendData(cid, {"type": "stop"})
        except Exception:
            pass

    summary_path = log_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "algorithm": "helcfl_real",
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "frac": args.frac,
        "local_ep": args.local_ep,
        "records": summary_records,
        "best_acc": best_acc,
        "final_acc": summary_records[-1]["acc"] if summary_records else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved summary to {summary_path}")
