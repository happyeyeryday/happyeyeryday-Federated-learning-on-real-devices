# HELCFL Real Deploy Quickstart

本文档说明如何把当前 `HELCFL real` 代码同步到 server 和所有 client，并启动实验。

## 1. 填设备表

先编辑本地文件：

- `deploy_sh/devices_local.py`

格式如下：

```python
SERVER_DEVICES = [
    ("192.168.31.105", "server_user", "server_password"),
]

CLIENT_DEVICES = [
    ("192.168.31.121", "jetsonnano", "admin"),
    ("192.168.31.244", "jetsonnano", "admin"),
    ("192.168.31.231", "jetsonnano", "admin"),
]

ALL_DEVICES = CLIENT_DEVICES + SERVER_DEVICES
```

注意：

- 每台物理设备只能对应一个 `CID`
- 不要把同一台机器重复写进多个 client 条目
- `devices_local.py` 是本地私有配置，不会提交到 Git

## 2. deploy 脚本做什么

当前 `HELCFL real` 的 deploy 会同步这些文件：

- server:
  - `server_helcfl_real.py`
  - `utils/helcfl_real_profiles.py`
  - `utils/power_manager_real.py`
  - `utils/options.py`
- client:
  - `client_helcfl_real.py`
  - `utils/helcfl_real_profiles.py`
  - `utils/power_manager_real.py`
  - `utils/options.py`

远端目录统一是：

- `~/server_fedavg`

## 3. 执行 deploy

在仓库根目录执行：

```bash
python3 deploy_sh/deploy_seed.py
python3 deploy_sh/deploy_server.py
python3 deploy_sh/deploy_cluster.py
```

说明：

- `deploy_seed.py` 把共享配置同步到所有设备
- `deploy_server.py` 只同步 server 需要的文件
- `deploy_cluster.py` 只同步所有 client 需要的文件

如果只想先试 server：

```bash
python3 deploy_sh/deploy_server.py
```

## 4. 启动实验

### server

在 server 上：

```bash
cd ~/server_fedavg
nohup python3 server_helcfl_real.py > server_helcfl_real.log 2>&1 &
```

### client

在每台 client 上：

```bash
cd ~/server_fedavg
nohup python3 client_helcfl_real.py --CID 0 > client_cid.log 2>&1 &
```

把 `--CID 0` 改成该设备实际对应的 `CID`。

## 5. 运行前检查

重点确认：

- `utils/options.py` 里的 `HOST` 是 server IP
- `utils/options.py` 里的 `num_users`、`frac` 符合本次实验
- 每台 client 的 `CID` 唯一且和设备表一致
- 设备具备免密 sudo，至少允许：
  - `nvpmodel`
  - `jetson_clocks`
  - `poweroff`

Nano/Xavier/Orin Nano 当前 mode 映射是：

- `nano: high=0, low=1`
- `agx_xavier: high=0, low=2`
- `orinnanosuper: low=0, high=2`

## 6. 日志怎么看

### client 正常开始训练

看到这些日志说明 client 正常工作：

- `DVFS switch confirmed ...`
- `Starting local training ...`
- `Finished epoch ...`
- `ack={'type': 'upload_ack', 'round': ...}`

### server 正常收到结果

server 端会看到：

- `ROUND=... selected=[...]`
- `RECV cid=...`
- `acc_model4=...`

## 7. 常见问题

### 只传到一台 client

检查 `deploy_sh/devices_local.py` 里的 `CLIENT_DEVICES`。
`deploy_cluster.py` 只会传给 `CLIENT_DEVICES` 中列出的机器。

### `Address already in use`

说明旧 server 还占着端口。先清理旧进程：

```bash
pkill -9 -f server_helcfl_real.py
```

### `DVFS mode switch failed: [Errno 12] Cannot allocate memory`

通常不是 sudo 权限问题，而是设备当时内存太紧，起外部命令子进程失败。先检查：

```bash
free -h
swapon --show
sudo -n true
sudo -n nvpmodel -m 1
sudo -n jetson_clocks
```
