import os
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from models.hetero_model import resnet18
from utils.ConnectHandler_client import ConnectHandler
from utils.FL_utils import DatasetSplit
from utils.get_dataset import get_dataset
from utils.helcfl_real_profiles import (
    DEVICE_DVFS_MODES,
    get_device_type,
)
from utils.options import args_parser
from utils.set_seed import set_random_seed


def set_jetson_mode(mode_id: int):
    start = time.strftime("%Y-%m-%d %H:%M:%S")
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
    end = time.strftime("%Y-%m-%d %H:%M:%S")
    return start, end


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)

    device_type = get_device_type(args.CID)
    log_dir = Path("logs_real/helcfl_real_client")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"client_helcfl_real_{args.CID}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.info(
        f"Starting HELCFL real client cid={args.CID} device_type={device_type} "
        f"gpu={args.gpu} modes={DEVICE_DVFS_MODES[device_type]}"
    )

    dataset_train, _, dict_users = get_dataset(args)
    connect_handler = ConnectHandler(args.HOST, args.POST, args.CID)
    loss_func = nn.CrossEntropyLoss()

    while True:
        recv = connect_handler.receiveFromServer()
        if not recv:
            logger.warning("Received empty payload, continue.")
            continue

        msg_type = recv.get("type")
        if msg_type == "stop":
            logger.info("Received stop signal from server.")
            break
        if msg_type != "train_round":
            logger.warning(f"Unknown message type: {msg_type}")
            continue

        round_idx = int(recv["round"])
        model_id = int(recv["model_id"])
        model_rate = float(recv["model_rate"])
        dvfs_mode = int(recv["dvfs_mode"])
        lr = float(recv.get("lr", args.lr))
        idxs_list = recv["idxs_list"]

        try:
            mode_switch_start, mode_switch_end = set_jetson_mode(dvfs_mode)
        except Exception as exc:
            logger.error(f"DVFS mode switch failed: {exc}")
            connect_handler.uploadToServer(
                {
                    "type": "client_error",
                    "round": round_idx,
                    "cid": args.CID,
                    "reason": f"dvfs_failed:{exc}",
                }
            )
            continue

        local_model = resnet18(model_rate=model_rate, track=False).to(args.device)
        local_model.load_state_dict(recv["net"])
        dt_loader = DataLoader(DatasetSplit(dataset_train, idxs_list), batch_size=args.bs, shuffle=True)
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        train_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        local_model.train()
        for _ in range(int(recv.get("local_ep", args.local_ep))):
            for images, labels in dt_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                logits = local_model(images)
                loss = loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        train_end_time = time.strftime("%Y-%m-%d %H:%M:%S")

        upload_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        connect_handler.uploadToServer(
            {
                "type": "client_update",
                "cid": args.CID,
                "round": round_idx,
                "model_id": model_id,
                "model_rate": model_rate,
                "dvfs_mode": dvfs_mode,
                "net": local_model.state_dict(),
                "num_samples": len(dict_users[args.CID]),
                "mode_switch_start_time": mode_switch_start,
                "mode_switch_end_time": mode_switch_end,
                "train_start_time": train_start_time,
                "train_end_time": train_end_time,
                "upload_start_time": upload_start_time,
                "upload_end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        ack = connect_handler.receiveFromServer()
        logger.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ROUND={round_idx} "
            f"cid={args.CID} device_type={device_type} model_id={model_id} "
            f"rate={model_rate} mode={dvfs_mode} ack={ack}"
        )
