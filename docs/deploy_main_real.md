# Main Real Deploy Quickstart

本文档说明如何把离线训练好的主方法策略包接入真实设备 `SHFL_resnet` 流程，并启动 `server_main_real.py`。

## 1. 准备策略包

先在仿真仓库重新导出一份新格式 bundle。bundle 目录里至少需要：

- `low_level_eval_rnn.pt`
- `high_level_role_eval.pt`
- `policy_manifest.json`

当前真实设备入口依赖 `policy_manifest.json`，旧的只有 `meta.txt` 的 bundle 不能直接用。

## 2. 同步代码

把下面这些文件同步到真实设备使用的代码目录：

- `server_main_real.py`
- `client_main_real.py`
- `utils/options.py`
- `utils/main_real_policy.py`
- `utils/main_real_profiles.py`
- `utils/power_manager_real.py`

## 3. 把 bundle 放到 server

把策略包目录拷到 server 上，例如：

```bash
scp -r /path/to/bundle server_user@server_ip:~/server_fedavg/policy_bundles/
```

最终假设 bundle 在：

```bash
~/server_fedavg/policy_bundles/main_bundle_xxx
```

## 4. 启动 server

在 server 上：

```bash
cd ~/server_fedavg
nohup python3 server_main_real.py \
  --policy_bundle ~/server_fedavg/policy_bundles/main_bundle_xxx \
  --policy_mode offline_bundle \
  > server_main_real.log 2>&1 &
```

说明：

- `--policy_bundle` 必填
- `--policy_manifest` 可省略，默认读取 bundle 目录里的 `policy_manifest.json`
- `--frac` 对 `main_real` 不再决定每轮选多少 client，真正的参与集合由策略输出的 `idle/train` 动作决定

## 5. 启动 client

主方法 client 使用 `client_main_real.py`：

```bash
cd ~/server_fedavg
nohup python3 client_main_real.py --CID 0 --log_tag main_real > client_cid0.log 2>&1 &
```

## 6. 行为说明

- server 每轮会为所有活跃 client 做一次离线策略推理
- 动作映射固定为：
  - `0..3 -> low dvfs + model_idx 1..4`
  - `4..7 -> high dvfs + model_idx 1..4`
  - `8 -> idle`
- 只有被判定为 `train` 的 client 才会收到 `train_round`
- 没被选中的 client，server 会按上一次 DVFS 档位估算 idle 电量下降

## 7. 日志检查

server 正常时会看到：

- `ROUND=... selected=[...] idle=[...]`
- `model_indices=[...] dvfs_modes=[...] roles=[...]`
- `acc_model4=...`

client 正常时会看到：

- `DVFS switch confirmed ...`
- `Starting local training ...`
- `ack={'type': 'upload_ack', 'round': ...}`
