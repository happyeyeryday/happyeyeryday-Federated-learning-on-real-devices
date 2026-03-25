import argparse
import pickle

import torch

from models.SHFL_resnet import shfl_resnet18


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
                if block_id <= target_exit_idx:
                    local_state[k] = v.clone()
            except Exception:
                pass
        elif k.startswith("bottlenecks"):
            try:
                exit_id = int(k.split(".")[1])
                if exit_id == target_exit_idx:
                    local_state["bottleneck." + ".".join(k.split(".")[2:])] = v.clone()
            except Exception:
                pass
        elif k.startswith("fcs"):
            try:
                exit_id = int(k.split(".")[1])
                if exit_id == target_exit_idx:
                    local_state["fc." + ".".join(k.split(".")[2:])] = v.clone()
            except Exception:
                pass
    return local_state


def format_bytes(size_bytes):
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def count_tensor_bytes(state_dict):
    total = 0
    for tensor in state_dict.values():
        total += tensor.numel() * tensor.element_size()
    return total


def estimate_sizes(model_idx, idxs_len, local_ep, lr):
    global_model = shfl_resnet18(num_classes=10)
    local_state = shfl_distribute(global_model.state_dict(), model_idx)
    dummy_idxs = list(range(idxs_len))

    train_round_payload = {
        "type": "train_round",
        "round": 0,
        "net": local_state,
        "idxs_list": dummy_idxs,
        "model_idx": model_idx,
        "model_depth_ratio": model_idx / 4.0,
        "dvfs_label": "low",
        "dvfs_mode": 1,
        "local_ep": local_ep,
        "lr": lr,
        "battery_joules": 150000.0,
        "battery_level": 1.0,
    }
    train_round_size = len(pickle.dumps(train_round_payload, protocol=pickle.HIGHEST_PROTOCOL))

    client_update_payload = {
        "type": "client_update",
        "cid": 0,
        "round": 0,
        "model_idx": model_idx,
        "model_depth_ratio": model_idx / 4.0,
        "dvfs_mode": 1,
        "net": local_state,
        "num_samples": idxs_len,
        "battery_joules": 149000.0,
        "battery_level": 0.9933,
        "mode_switch_start_time": "2026-03-23 16:00:00",
        "mode_switch_end_time": "2026-03-23 16:00:01",
        "train_start_time": "2026-03-23 16:00:01",
        "train_end_time": "2026-03-23 16:03:01",
        "upload_start_time": "2026-03-23 16:03:01",
        "upload_end_time": "2026-03-23 16:03:03",
    }
    client_update_size = len(pickle.dumps(client_update_payload, protocol=pickle.HIGHEST_PROTOCOL))

    return {
        "model_idx": model_idx,
        "tensor_bytes": count_tensor_bytes(local_state),
        "train_round_bytes": train_round_size,
        "client_update_bytes": client_update_size,
        "tensor_keys": len(local_state),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate HELCFL real payload sizes by model_idx.")
    parser.add_argument("--idxs-len", type=int, default=2000, help="Length of idxs_list in train_round payload.")
    parser.add_argument("--local-ep", type=int, default=5, help="local_ep value placed in payload.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate value placed in payload.")
    parser.add_argument(
        "--model-idxs",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4],
        help="Model indices to estimate.",
    )
    args = parser.parse_args()

    print(f"idxs_len={args.idxs_len} local_ep={args.local_ep} lr={args.lr}")
    print(
        "model_idx | state_dict | train_round(full) | client_update(full) | tensor_keys"
    )
    print("-" * 78)
    for model_idx in args.model_idxs:
        result = estimate_sizes(model_idx, args.idxs_len, args.local_ep, args.lr)
        print(
            f"{result['model_idx']:>9} | "
            f"{format_bytes(result['tensor_bytes']):>10} | "
            f"{format_bytes(result['train_round_bytes']):>17} | "
            f"{format_bytes(result['client_update_bytes']):>19} | "
            f"{result['tensor_keys']:>11}"
        )


if __name__ == "__main__":
    main()
