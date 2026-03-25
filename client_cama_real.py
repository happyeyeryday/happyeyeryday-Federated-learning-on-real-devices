import subprocess
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.SHFL_resnet import shfl_resnet18
from utils.ConnectHandler_client import ConnectHandler
from utils.cama_real_profiles import DEVICE_DVFS_MODES, get_device_type
from utils.get_dataset import DatasetSplit, get_dataset
from utils.options import args_parser
from utils.power_manager_real import BatteryManagerReal, LOW_BATTERY_THRESHOLD_J
from utils.set_seed import set_random_seed

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_str():
    return time.strftime(TIME_FORMAT)


def set_jetson_mode(mode_id):
    start = now_str()
    result = subprocess.run(
        ["sudo", "nvpmodel", "-m", str(mode_id)],
        input="NO\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output = result.stdout + result.stderr
    if "Reboot required" in output or "REBOOT" in output.upper():
        raise RuntimeError(f"mode {mode_id} requires reboot")

    subprocess.run(["sudo", "jetson_clocks"], check=True)
    try:
        subprocess.run(
            "echo 255 | sudo tee /sys/devices/virtual/thermal/cooling_device3/cur_state",
            shell=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    end = now_str()
    return start, end


def query_jetson_mode():
    result = subprocess.run(
        ["sudo", "nvpmodel", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


def build_local_model(args, model_idx):
    return shfl_resnet18(num_classes=args.num_classes, model_idx=model_idx).to(args.device)


def maybe_sync_battery(battery_manager, payload):
    if "battery_joules" in payload:
        battery_manager.set_charge(payload["battery_joules"])
        return True
    if "battery_level" in payload:
        battery_manager.set_charge(float(payload["battery_level"]) * battery_manager.total_capacity)
        return True
    if "battery_ratio" in payload:
        battery_manager.set_charge(float(payload["battery_ratio"]) * battery_manager.total_capacity)
        return True
    return False


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)

    device_type = get_device_type(args.CID)
    battery_manager = BatteryManagerReal(device_type=device_type)
    log_dir = Path("logs_real/cama_real_client")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"client_cama_real_{args.CID}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.info(
        f"Starting CAMA real client cid={args.CID} device_type={device_type} "
        f"gpu={args.gpu} modes={DEVICE_DVFS_MODES[device_type]}"
    )

    dataset_train, _, dict_users = get_dataset(args)
    connect_handler = ConnectHandler(args.HOST, args.POST, args.CID)
    loss_func = nn.CrossEntropyLoss()
    battery_state_synced = False
    last_activity_ts = time.time()

    while True:
        idle_start = time.time()
        idle_duration = idle_start - last_activity_ts
        if battery_state_synced and idle_duration > 0:
            battery_manager.consume("idle", idle_duration)

        recv_start = time.time()
        recv = connect_handler.receiveFromServer()
        recv_duration = time.time() - recv_start

        if not recv:
            logger.warning("Received empty payload, continue.")
            last_activity_ts = time.time()
            continue

        if maybe_sync_battery(battery_manager, recv):
            battery_state_synced = True
        if battery_state_synced:
            battery_manager.consume("communication", recv_duration)

        msg_type = recv.get("type")
        if msg_type == "stop":
            logger.info("Received stop signal from server.")
            break
        if msg_type != "train_round":
            logger.warning(f"Unknown message type: {msg_type}")
            last_activity_ts = time.time()
            continue

        round_idx = int(recv["round"])
        model_idx = int(recv["model_idx"])
        model_depth_ratio = float(recv.get("model_depth_ratio", model_idx / 4.0))
        dvfs_mode = int(recv["dvfs_mode"])
        lr = float(recv.get("lr", args.lr))
        idxs_list = recv["idxs_list"]
        mode_label = battery_manager.set_power_mode_by_mode_id(dvfs_mode)

        if not battery_manager.check_energy(LOW_BATTERY_THRESHOLD_J):
            logger.warning(f"cid={args.CID} battery too low before training, shutting down.")
            status_upload_start = time.time()
            connect_handler.uploadToServer(
                {
                    "type": "status",
                    "status": "low_battery",
                    "cid": args.CID,
                    "round": round_idx,
                    "device_type": device_type,
                    "dvfs_mode": dvfs_mode,
                    "battery_joules": battery_manager.get_charge(),
                    "battery_level": battery_manager.get_ratio(),
                }
            )
            battery_manager.consume("communication", time.time() - status_upload_start, mode_label=mode_label)
            ack = connect_handler.receiveFromServer()
            logger.info(f"Shutdown ack={ack}")
            subprocess.run(["sudo", "poweroff"], check=False)
            break

        try:
            mode_switch_start, mode_switch_end = set_jetson_mode(dvfs_mode)
        except Exception as exc:
            logger.error(f"DVFS mode switch failed: {exc}")
            error_upload_start = time.time()
            connect_handler.uploadToServer(
                {
                    "type": "client_error",
                    "round": round_idx,
                    "cid": args.CID,
                    "device_type": device_type,
                    "status": "client_error",
                    "failure_reason": f"dvfs_failed:{exc}",
                    "reason": f"dvfs_failed:{exc}",
                    "battery_joules": battery_manager.get_charge(),
                    "battery_level": battery_manager.get_ratio(),
                }
            )
            battery_manager.consume("communication", time.time() - error_upload_start, mode_label=mode_label)
            last_activity_ts = time.time()
            continue

        try:
            current_mode_info = query_jetson_mode()
            logger.info(
                f"DVFS switch confirmed for cid={args.CID}: "
                f"target_mode={dvfs_mode} mode_label={mode_label} query='{current_mode_info}'"
            )
        except Exception as exc:
            logger.warning(
                f"DVFS verification query failed for cid={args.CID}: "
                f"target_mode={dvfs_mode} error={exc}"
            )

        local_model = build_local_model(args, model_idx)
        local_model.load_state_dict(recv["net"], strict=False)
        dt_loader = DataLoader(
            DatasetSplit(dataset_train, idxs_list),
            batch_size=args.local_bs,
            shuffle=True,
        )
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        train_start_time = now_str()
        train_start_ts = time.time()
        local_sample_count = len(idxs_list)
        logger.info(
            f"Starting local training cid={args.CID} round={round_idx} "
            f"model_idx={model_idx} mode={dvfs_mode} mode_label={mode_label} "
            f"local_samples={local_sample_count} batches_per_epoch={len(dt_loader)}"
        )
        local_model.train()
        total_epochs = int(recv.get("local_ep", args.local_ep))
        for epoch in range(total_epochs):
            progress = tqdm(
                dt_loader,
                desc=f"CID {args.CID} Round {round_idx} Epoch {epoch + 1}/{total_epochs}",
                unit="batch",
                file=sys.stdout,
            )
            for images, labels in progress:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                logits = local_model(images)
                loss = loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            logger.info(
                f"Finished epoch cid={args.CID} round={round_idx} "
                f"epoch={epoch + 1}/{total_epochs}"
            )
        train_duration = time.time() - train_start_ts
        train_end_time = now_str()
        battery_manager.consume("train", train_duration, mode_label=mode_label)

        upload_start_time = now_str()
        upload_start_ts = time.time()
        connect_handler.uploadToServer(
            {
                "type": "client_update",
                "cid": args.CID,
                "device_type": device_type,
                "status": "ok",
                "failure_reason": "",
                "round": round_idx,
                "model_idx": model_idx,
                "model_depth_ratio": model_depth_ratio,
                "dvfs_mode": dvfs_mode,
                "net": local_model.state_dict(),
                "num_samples": len(dict_users[args.CID]),
                "battery_joules": battery_manager.get_charge(),
                "battery_level": battery_manager.get_ratio(),
                "mode_switch_start_time": mode_switch_start,
                "mode_switch_end_time": mode_switch_end,
                "train_start_time": train_start_time,
                "train_end_time": train_end_time,
                "upload_start_time": upload_start_time,
                "upload_end_time": now_str(),
            }
        )
        ack = connect_handler.receiveFromServer()
        upload_duration = time.time() - upload_start_ts
        battery_manager.consume("communication", upload_duration, mode_label=mode_label)
        logger.info(
            f"{now_str()} ROUND={round_idx} cid={args.CID} device_type={device_type} "
            f"model_idx={model_idx} depth_ratio={model_depth_ratio:.2f} "
            f"mode={dvfs_mode} mode_label={mode_label} "
            f"battery={battery_manager.get_charge():.2f}J ack={ack}"
        )
        last_activity_ts = time.time()
