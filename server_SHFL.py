import time
import copy
import torch
from torch import nn
import numpy as np
import random
from loguru import logger
from torch.utils.data import DataLoader
import datetime
import os

# 引入工具类
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from utils.power_manager import BATTERY_CAPACITY, POWER_CONFIG, wake_clients

# [新增] 引入 SHFL 智能体
from utils.shfl_agent import SHFLAgentManager

from models.SHFL_resnet import shfl_resnet18

# 断点恢复时，如果存在 sleeping 客户端，给它们额外时间完成 WoL 配置并进入 suspend。
SYNC_SLEEP_PREPARE_TIME = int(os.getenv("SYNC_SLEEP_PREPARE_TIME", "10"))
# 对 Orin 的兼容策略：断点恢复时仅恢复电量，不恢复 runtime_state（避免恢复阶段 sleep/wake 时序异常）。
ORIN_SKIP_RUNTIME_RESTORE = os.getenv("ORIN_SKIP_RUNTIME_RESTORE", "1") == "1"

# 全局变量
net_glob = None
w_local_client = []

# =============================================================================
# 1. SHFL 专用分发函数 (适配 BYOT 结构: mainblocks, bottlenecks, fcs)
# =============================================================================
def shfl_distribute(global_model_state, model_idx):
    """
    从Server全局模型(Three_ResNet_ALL)切片下发到Client(Three_ResNet)。

    关键：Server有bottlenecks[0-3]和fcs[0-3]，Client只有bottleneck和fc
    需要将bottlenecks.{model_idx-1}.xxx重命名为bottleneck.xxx

    Args:
        model_idx: 1 (Block0), 2 (Block0+1), 3 (Block0-2), 4 (Full)

    Structure to Keep:
    - conv1, bn1: Always
    - mainblocks.i: if i <= target_idx
    - bottlenecks.target_idx -> bottleneck (重命名)
    - fcs.target_idx -> fc (重命名)
    """
    local_state = {}
    target_exit_idx = model_idx - 1  # model_idx 1 -> exit_idx 0

    for k, v in global_model_state.items():
        # 过滤 BN 统计量 (sBN)
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue

        # 1. 公共头部 (conv1, bn1) - 所有人都要
        if k.startswith('conv1') or k.startswith('bn1'):
            local_state[k] = v.clone()

        # 2. 主干网络 (mainblocks)
        elif k.startswith('mainblocks'):
            # 格式: mainblocks.0.xxx
            try:
                parts = k.split('.')
                block_id = int(parts[1])
                # 只发送 <= target_exit_idx 的块
                if block_id <= target_exit_idx:
                    local_state[k] = v.clone()
            except: pass

        # 3. 瓶颈层 (bottlenecks.i -> bottleneck)
        elif k.startswith('bottlenecks'):
            # 格式: bottlenecks.0.conv1.weight
            try:
                parts = k.split('.')
                exit_id = int(parts[1])
                # 只取当前模型对应的exit
                if exit_id == target_exit_idx:
                    # 重命名: bottlenecks.0.xxx -> bottleneck.xxx
                    new_key = 'bottleneck.' + '.'.join(parts[2:])
                    local_state[new_key] = v.clone()
            except: pass

        # 4. 分类头 (fcs.i -> fc)
        elif k.startswith('fcs'):
            # 格式: fcs.0.weight
            try:
                parts = k.split('.')
                exit_id = int(parts[1])
                # 只取当前模型对应的exit
                if exit_id == target_exit_idx:
                    # 重命名: fcs.0.weight -> fc.weight
                    new_key = 'fc.' + '.'.join(parts[2:])
                    local_state[new_key] = v.clone()
            except: pass

    return local_state

# =============================================================================
# 2. SHFL 聚合函数
# =============================================================================
def shfl_aggregate(w_local_list, model_indices, net_glob):
    """
    聚合不同深度的模型到Server全局模型(Three_ResNet_ALL)。

    关键：Client返回的是bottleneck.xxx和fc.xxx，需要重命名为bottlenecks.{exit_idx}.xxx
    不同深度的Client更新不同的exit，互不冲突。

    Args:
        w_local_list: Client参数列表
        model_indices: 每个Client使用的model_idx (1-4)
        net_glob: Server全局模型
    """
    global_state = net_glob.state_dict()
    sum_buffer = {}
    count_buffer = {}

    # 初始化 Buffer
    for k, v in global_state.items():
        if 'num_batches_tracked' in k: continue
        sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
        count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    for idx, local_w in enumerate(w_local_list):
        model_idx = model_indices[idx]
        exit_idx = model_idx - 1  # 1->0, 2->1, 3->2, 4->3

        for k_local, v_local in local_w.items():
            # 反向重命名: bottleneck.xxx -> bottlenecks.{exit_idx}.xxx
            if k_local.startswith('bottleneck.'):
                suffix = k_local[len('bottleneck.'):]  # 去掉前缀
                k_global = f'bottlenecks.{exit_idx}.{suffix}'
            # 反向重命名: fc.xxx -> fcs.{exit_idx}.xxx
            elif k_local.startswith('fc.'):
                suffix = k_local[len('fc.'):]
                k_global = f'fcs.{exit_idx}.{suffix}'
            # 其他key保持不变 (conv1, bn1, mainblocks)
            else:
                k_global = k_local

            if k_global not in sum_buffer:
                continue

            # 累加到对应的全局参数
            sum_buffer[k_global] += v_local
            count_buffer[k_global] += 1

    # 平均并更新
    updated_state = {}
    for k in global_state.keys():
        if 'num_batches_tracked' in k:
            updated_state[k] = global_state[k]  # BN统计量不聚合
            continue

        mask = count_buffer[k] > 0
        updated_state[k] = global_state[k].clone()

        if mask.any():
            updated_state[k][mask] = (sum_buffer[k][mask] / count_buffer[k][mask]).to(global_state[k].dtype)

    return updated_state

# =============================================================================
# 3. BN 校准 & 评估 (适配 BYOT)
# =============================================================================
def stats(dataset, model, args):
    logger.info("Starting BN calibration...")
    # 打印数据集大小，检查 Server 是否真的持有数据
    logger.info(f"DEBUG: Server dataset size: {len(dataset)}")
    test_model = copy.deepcopy(model)
    test_model.to(args.device)

    # 打印校准前的 BN 均值（抽查第一层）
    for name, module in test_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            logger.info(f"DEBUG: Before Calib - {name} mean head: {module.running_mean[:5].tolist()}")
            break

    # 强制所有 BN 层进入追踪模式并重置统计量
    test_model.train()  # 🔥 关键：必须设置为train模式
    for module in test_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()  # 重置统计量
            module.track_running_stats = True
            module.momentum = None  # 🔥 关键：使用累积平均而非指数移动平均

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    with torch.no_grad():  # 不需要梯度，但BN统计量仍会更新
        for i, (images, _) in enumerate(data_loader):
            if i >= 50: break
            images = images.to(args.device)
            # BYOT Forward 返回 tuple/list，跑通即可
            _ = test_model(images)

    # 打印校准后的 BN 均值
    for name, module in test_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            logger.info(f"DEBUG: After Calib  - {name} mean head: {module.running_mean[:5].tolist()}")
            break

    test_model.eval()
    logger.info("✅ BN calibration finished.")
    return test_model

def summary_evaluate(net, dataset_test, device):
    """
    评估所有 4 个模型的准确率
    Returns:
        list: [acc_model1, acc_model2, acc_model3, acc_model4] (百分比形式)
    """
    net.eval()
    dtLoader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)

    # 初始化 4 个模型的累加器
    num_models = 4
    metrics = [Accumulator(2) for _ in range(num_models)]

    with torch.no_grad():
        for images, labels in dtLoader:
            images, labels = images.to(device), labels.to(device)
            # Three_ResNet_ALL Forward: (output_list, features_end, embedding)
            outputs = net(images)

            # Three_ResNet_ALL返回tuple: (output_list[0-3], features_end, embedding)
            if isinstance(outputs, tuple):
                logits_list = outputs[0]  # [model1_out, model2_out, model3_out, model4_out]
            else:
                # 兼容Three_ResNet（不应该走到这里）
                logits_list = [outputs]

            # 计算每个模型的准确率
            for i, pred in enumerate(logits_list):
                metrics[i].add(accuracy(pred, labels), labels.numel())

    # 返回所有准确率（转为百分比）
    acc_list = [(m[0] / m[1]) * 100 for m in metrics]
    return acc_list

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 1. 设备配置
    device_info_map = {
        0: {'mac': "48:b0:2d:3c:cc:ff", 'ip': "192.168.31.161"}, # Nano
        1: {'mac': "48:b0:2d:3c:ca:18", 'ip': "192.168.31.239"}, # Nano
        2: {'mac': "48:b0:2d:3c:cc:f3", 'ip': "192.168.31.142"}, # Nano
        3: {'mac': "48:b0:2d:3c:cc:df", 'ip': "192.168.31.121"}, # Nano
        4: {'mac': "48:b0:2d:3c:ca:27", 'ip': "192.168.31.237"}, # Nano
        5: {'mac': "48:b0:2d:3c:ca:20", 'ip': "192.168.31.231"}, # Nano
        6: {'mac': "48:b0:2d:3c:ca:2e", 'ip': "192.168.31.244"}, # Nano
        7: {'mac': "48:b0:2d:2f:e2:d9", 'ip': "192.168.31.243"}, # Xavier
        8: {'mac': "48:b0:2d:2f:eb:0e", 'ip': "192.168.31.198"}, # Xavier
        9: {'mac': "3c:6d:66:28:57:90", 'ip': "192.168.31.19"}, # Orin
    }

    num_physical_clients = len(device_info_map)
    active_clients = list(range(num_physical_clients))

    # 2. 硬件能力配置 (C)
    COMPUTE_CAPABILITY = {
        'orin': 1.0,
        'xavier': 0.55,
        'nano': 0.076
    }

    # 3. 初始化 Client 状态
    client_states = {}
    # 定义 ID 到类型的映射，方便查表
    # 0=Orin, 1-2=Xavier, 3-9=Nano
    id_to_type = {}
    for i in range(args.num_users):
        if i == 9: t = 'orin'
        elif i == 8 or i == 7: t = 'xavier'
        else: t = 'nano'
        id_to_type[i] = t

        client_states[i] = {
            'C': COMPUTE_CAPABILITY[t],
            'E': 1.0,
            'L': 0.0 # 稍后填充
        }

    client_battery_joules = {
        i: float(BATTERY_CAPACITY[id_to_type[i]])
        for i in range(num_physical_clients)
    }
    retired_clients = set()
    client_runtime_state = {
        i: 'waiting'
        for i in range(num_physical_clients)
    }

    def get_device_capacity(client_id):
        return float(BATTERY_CAPACITY[id_to_type[client_id]])

    def normalize_battery(client_id, joules):
        capacity = get_device_capacity(client_id)
        joules = min(max(float(joules), 0.0), capacity)
        return joules / capacity

    def restore_battery_from_checkpoint(checkpoint, client_id):
        if 'battery_state_joules' in checkpoint and client_id in checkpoint['battery_state_joules']:
            restored = checkpoint['battery_state_joules'][client_id]
        elif (
            'client_states' in checkpoint
            and client_id in checkpoint['client_states']
            and 'E' in checkpoint['client_states'][client_id]
        ):
            legacy_ratio = min(max(float(checkpoint['client_states'][client_id]['E']), 0.0), 1.0)
            restored = legacy_ratio * get_device_capacity(client_id)
        else:
            restored = get_device_capacity(client_id)
        capacity = get_device_capacity(client_id)
        return min(max(float(restored), 0.0), capacity)

    def build_runtime_state_snapshot(sleeping_clients=None):
        sleeping_clients = sleeping_clients or set()
        snapshot = {}
        for client_id in range(num_physical_clients):
            if client_id in retired_clients:
                snapshot[client_id] = 'retired'
            elif client_id in sleeping_clients:
                snapshot[client_id] = 'sleeping'
            else:
                snapshot[client_id] = 'waiting'
        return snapshot

    # ================= [初始化] =================
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # 填充 L
    max_data_len = max([len(dict_users[i]) for i in range(num_physical_clients)])
    for idx in client_states:
        client_states[idx]['L'] = len(dict_users[idx]) / max_data_len

    logger.info("Initializing SHFL Global Model (BYOT)...")
    # [🔥 修正] 使用 shfl_resnet18 工厂函数
    net_glob = shfl_resnet18(num_classes=args.num_classes)
    net_glob.apply(init_weights)
    net_glob.to(args.device)

    # [DEBUG] 打印模型的key结构
    logger.info("=" * 60)
    logger.info("Global Model Keys:")
    for k in list(net_glob.state_dict().keys())[:10]:
        logger.info(f"  {k}")
    logger.info("  ...")
    for k in list(net_glob.state_dict().keys())[-5:]:
        logger.info(f"  {k}")
    logger.info("=" * 60)

    # ================= [RL Agent 初始化] =================
    try:
        rl_agent = SHFLAgentManager(n_agents=num_physical_clients, model_dir='./SHFL_model/')
        logger.success("RL Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to init RL Agent: {e}")
        exit(1)

    summary_acc_test_collect = []
    time_collect = []

    logger.info(f"🔌 Server listening for {num_physical_clients} clients...")
    connectHandler = ConnectHandler(num_physical_clients, args.HOST, args.POST)

    target_m = max(int(args.frac * num_physical_clients), 1)

    # ==========================================
    # [新增] 断点续训 - 加载 Checkpoint
    # ==========================================
    checkpoint_path = "checkpoint_shfl.pth"
    start_round = 0
    checkpoint = None

    if os.path.exists(checkpoint_path):
        logger.info(f"📂 Found checkpoint: {checkpoint_path}, Resuming...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            net_glob.load_state_dict(checkpoint['model_state_dict'])
            summary_acc_test_collect = checkpoint['acc_history']
            start_round = checkpoint['round'] + 1

            # [SHFL特有] 恢复client状态和电量
            if 'active_clients' in checkpoint:
                active_clients = checkpoint['active_clients']
                logger.info(f"  Restored active_clients: {active_clients}")
            for idx in range(num_physical_clients):
                restored_j = restore_battery_from_checkpoint(checkpoint, idx)
                client_battery_joules[idx] = restored_j
                client_states[idx]['E'] = normalize_battery(idx, restored_j)
            logger.info("  Restored client battery levels")

            # [SHFL特有] 恢复RL Agent状态
            if 'rl_agent_state' in checkpoint:
                try:
                    rl_agent.load_state(checkpoint['rl_agent_state'])
                    logger.info(f"  Restored RL Agent state")
                except Exception as e:
                    logger.warning(f"  Failed to restore RL Agent: {e}. Will retrain.")

            logger.success(f"✅ Successfully resumed from Round {start_round}!")
            last_acc = summary_acc_test_collect[-1]
            if isinstance(last_acc, list):
                logger.info(f"  Last Acc: Model1={last_acc[0]:.2f}%, Model2={last_acc[1]:.2f}%, Model3={last_acc[2]:.2f}%, Model4={last_acc[3]:.2f}%")
            else:
                logger.info(f"  Last Acc (Legacy): {last_acc:.2f}%")
        except Exception as e:
            logger.error(f"❌ Failed to resume: {e}. Starting from scratch.")
            start_round = 0
            checkpoint = None
    else:
        logger.info("🆕 No checkpoint found. Starting from scratch.")

    if checkpoint is not None:
        if 'retired_clients' in checkpoint:
            retired_clients = set(checkpoint['retired_clients'])
        else:
            for idx in range(num_physical_clients):
                if idx not in active_clients and client_battery_joules[idx] <= 50.0:
                    retired_clients.add(idx)
        active_clients = [idx for idx in active_clients if idx not in retired_clients]

        if 'client_runtime_state' in checkpoint:
            raw_runtime_state = checkpoint['client_runtime_state']
            client_runtime_state = {
                idx: raw_runtime_state.get(idx, 'waiting')
                for idx in range(num_physical_clients)
            }
        else:
            client_runtime_state = build_runtime_state_snapshot()

        checkpoint_saved_at = checkpoint.get('checkpoint_saved_at')
        if checkpoint_saved_at is not None:
            downtime = max(0.0, time.time() - float(checkpoint_saved_at))
            if downtime > 0:
                for idx, runtime_state in client_runtime_state.items():
                    if runtime_state == 'sleeping':
                        sleep_power = POWER_CONFIG[id_to_type[idx]]['sleep']
                        client_battery_joules[idx] = max(0.0, client_battery_joules[idx] - sleep_power * downtime)
                        client_states[idx]['E'] = normalize_battery(idx, client_battery_joules[idx])
                        if client_battery_joules[idx] <= 50.0:
                            retired_clients.add(idx)
                            if idx in active_clients:
                                active_clients.remove(idx)
                            client_runtime_state[idx] = 'retired'
                logger.info(f"  Applied {downtime:.1f}s sleep drain to sleeping clients")

        # 兜底校验：无论 checkpoint 中 active/retired 如何记录，低电量设备都不再参与训练。
        for idx in range(num_physical_clients):
            if client_battery_joules[idx] <= 50.0:
                retired_clients.add(idx)
                client_runtime_state[idx] = 'retired'
        active_clients = [idx for idx in active_clients if idx not in retired_clients]

    client_runtime_state = build_runtime_state_snapshot(
        {
            idx for idx, state in client_runtime_state.items()
            if state == 'sleeping' and idx not in retired_clients
        }
    )

    if ORIN_SKIP_RUNTIME_RESTORE:
        for idx in range(num_physical_clients):
            if id_to_type[idx] == 'orin' and idx not in retired_clients:
                if client_runtime_state.get(idx) != 'waiting':
                    logger.warning(
                        f"Applying Orin resume override for Client {idx}: "
                        f"{client_runtime_state.get(idx)} -> waiting"
                    )
                client_runtime_state[idx] = 'waiting'

    for idx in range(num_physical_clients):
        logger.info(
            f"🔋 [ResumeSync] Client {idx}: "
            f"{client_battery_joules[idx]:.2f} J "
            f"({normalize_battery(idx, client_battery_joules[idx]) * 100:.2f}%), "
            f"state={client_runtime_state[idx]}"
        )
        sync_msg = {
            'type': 'sync_state',
            'battery_joules': client_battery_joules[idx],
            'battery_level': normalize_battery(idx, client_battery_joules[idx]),
            'runtime_state': client_runtime_state[idx],
        }
        if not connectHandler.sendData(idx, sync_msg):
            logger.warning(f"⚠️ Failed to sync state to Client {idx}. Removing from active pool.")
            if idx in active_clients:
                active_clients.remove(idx)

    sleeping_clients_after_resume = [
        idx for idx, state in client_runtime_state.items() if state == 'sleeping'
    ]
    if sleeping_clients_after_resume:
        logger.info(
            f"⏸️ Detected {len(sleeping_clients_after_resume)} sleeping clients in resumed state. "
            f"Waiting {SYNC_SLEEP_PREPARE_TIME}s for clients to finish sleep preparation..."
        )
        time.sleep(SYNC_SLEEP_PREPARE_TIME)

    # ================= [训练循环] =================
    for iter in range(start_round, args.epochs):
        # 记录本轮开始时间
        round_start_time = datetime.datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"🕐 Round {iter} Start Time: {round_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logger.info(f"{'='*60}")

        if not active_clients:
            logger.critical("🪫 All clients have run out of battery.")
            break

        logger.info(f"🔋 Active Clients Pool: {len(active_clients)}")
        current_m = min(target_m, len(active_clients))

        # 1. 准备 RL 输入
        observations = []
        device_types = []
        for i in range(num_physical_clients):
            if i in active_clients:
                observations.append(client_states[i])
            else:
                # 死亡设备
                dead = client_states[i].copy()
                dead['E'] = 0.0
                observations.append(dead)
            # 添加设备类型信息
            device_types.append(id_to_type[i])

        # 2. RL 决策 (传入设备类型)
        decisions = rl_agent.select_models(observations, device_types, round_num=iter)

        # 3. Top-K 筛选
        valid_decisions = [d for d in decisions if d['client_idx'] in active_clients]
        valid_decisions.sort(key=lambda x: x['q_value'], reverse=True)
        top_k = valid_decisions[:current_m]

        idxs_users = []
        client_model_map = {}
        for d in top_k:
            c_idx = d['client_idx']
            idxs_users.append(c_idx)
            client_model_map[c_idx] = d['action'] # Model 1-4

        idxs_users = np.array(idxs_users)

        print(f"\n{'='*60}")
        print(f"Round {iter}: RL Top-{current_m}")
        print(f"Selection: {idxs_users}")
        print(f"Models: {[client_model_map[i] for i in idxs_users]}")
        print(f"{'='*60}\n")

        # 4. 唤醒
        mac_to_ip_to_wake = {
            device_info_map[idx]['mac']: device_info_map[idx]['ip']
            for idx in idxs_users
        }
        wake_clients(mac_to_ip_to_wake, total_timeout=15, settle_time=8)

        # 5. 下发
        successful_indices = []
        for idx in idxs_users:
            model_idx = client_model_map[idx]

            msg = dict()
            # [🔥 修正] 使用 shfl_distribute
            w_local_slice = shfl_distribute(net_glob.state_dict(), model_idx)

            msg['net'] = w_local_slice
            msg['model_idx'] = model_idx
            msg['idxs_list'] = dict_users[idx]
            msg['type'] = 'net'
            msg['round'] = iter
            msg['battery_joules'] = client_battery_joules[idx]
            msg['battery_ratio'] = client_states[idx]['E']

            logger.info(f"📤 Send to Client {idx} (Model-{model_idx})")

            if connectHandler.sendData(idx, msg):
                successful_indices.append(idx)
            else:
                logger.error(f"❌ Failed to send to Client {idx}. Removing.")
                if idx in active_clients: active_clients.remove(idx)

        # 6. 接收
        w_local_client = []
        model_indices_this_round = []
        sleeping_clients_this_round = set()

        responses_received = 0
        expected_responses = len(successful_indices)

        while responses_received < expected_responses:
            try:
                msg, client_idx = connectHandler.receiveData()
            except Exception as e:
                logger.error(f"Error receiving: {e}")
                responses_received += 1
                continue
            responses_received += 1

            if 'battery_joules' in msg:
                client_battery_joules[client_idx] = min(
                    max(float(msg['battery_joules']), 0.0),
                    get_device_capacity(client_idx)
                )
                client_states[client_idx]['E'] = normalize_battery(client_idx, client_battery_joules[client_idx])
                logger.info(
                    f"🔋 Client {client_idx} battery update: "
                    f"{client_battery_joules[client_idx]:.2f} J "
                    f"({client_states[client_idx]['E'] * 100:.2f}%)"
                )
            elif 'battery_level' in msg:
                ratio = min(max(float(msg['battery_level']), 0.0), 1.0)
                client_battery_joules[client_idx] = ratio * get_device_capacity(client_idx)
                client_states[client_idx]['E'] = ratio
                logger.info(
                    f"🔋 Client {client_idx} battery update: "
                    f"{client_battery_joules[client_idx]:.2f} J "
                    f"({client_states[client_idx]['E'] * 100:.2f}%)"
                )

            if msg['type'] == 'status' and msg.get('status') == 'low_battery':
                logger.warning(f"🪫 Client {client_idx} Low Battery.")
                retired_clients.add(client_idx)
                connectHandler.sendData(client_idx, {'type': 'shutdown_ack'})
                if client_idx in active_clients:
                    active_clients.remove(client_idx)

            elif msg['type'] == 'net':
                logger.info(f"📥 Received from Client {client_idx}")
                w_local_client.append(msg['net'])
                model_indices_this_round.append(client_model_map[client_idx])
                sleeping_clients_this_round.add(client_idx)
                connectHandler.sendData(client_idx, {'type': 'upload_ack'})

        # 7. 聚合
        if len(w_local_client) > 0:
            logger.info(f"⚙️ Aggregating {len(w_local_client)} models...")
            # [🔥 修正] 使用 shfl_aggregate
            w_glob = shfl_aggregate(w_local_client, model_indices_this_round, net_glob)
            net_glob.load_state_dict(w_glob, strict=False)

        # 8. 校准与评估
        net_glob_calibrated = stats(dataset_train, net_glob, args)
        net_glob.load_state_dict(net_glob_calibrated.state_dict(), strict=False)

        # 评估所有模型
        acc_list = summary_evaluate(net_glob, dataset_test, args.device)  # [acc1, acc2, acc3, acc4]
        summary_acc_test_collect.append(acc_list)

        # 记录本轮结束时间和持续时长
        round_end_time = datetime.datetime.now()
        round_duration = (round_end_time - round_start_time).total_seconds()

        print("\n" + "="*60)
        print(f"📊 Round {iter} Accuracy Results:")
        print(f"   Model 1 (After Block 0): {acc_list[0]:.2f}%")
        print(f"   Model 2 (After Block 1): {acc_list[1]:.2f}%")
        print(f"   Model 3 (After Block 2): {acc_list[2]:.2f}%")
        print(f"   Model 4 (After Block 3): {acc_list[3]:.2f}%")
        print(f"🕐 Round Duration: {round_duration:.2f}s")
        print("="*60 + "\n")

        # 输出CSV格式数据，方便后续分析
        logger.info(f"RESULT_CSV,{iter},{acc_list[0]:.2f},{acc_list[1]:.2f},{acc_list[2]:.2f},{acc_list[3]:.2f},"
                   f"{round_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},"
                   f"{round_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{round_duration:.2f}")

        client_runtime_state = build_runtime_state_snapshot(sleeping_clients_this_round)

        # ==========================================
        # [新增] 断点续训 - 保存 Checkpoint
        # ==========================================
        checkpoint_path = "checkpoint_shfl.pth"
        checkpoint_path_tmp = checkpoint_path + ".tmp"  # 先写临时文件，防止保存时崩溃导致文件损坏

        state = {
            'round': iter,
            'model_state_dict': net_glob.state_dict(),
            'acc_history': summary_acc_test_collect,
            # [SHFL特有] 保存client状态和活跃列表
            'active_clients': active_clients,
            'client_states': client_states,
            'battery_state_joules': client_battery_joules,
            'battery_state_ratio': {
                idx: normalize_battery(idx, client_battery_joules[idx])
                for idx in client_battery_joules
            },
            'retired_clients': sorted(retired_clients),
            'client_runtime_state': client_runtime_state,
            'checkpoint_saved_at': time.time(),
            # [SHFL特有] 保存RL Agent状态
            'rl_agent_state': rl_agent.get_state() if hasattr(rl_agent, 'get_state') else None,
        }

        try:
            torch.save(state, checkpoint_path_tmp)
            os.replace(checkpoint_path_tmp, checkpoint_path)  # 原子操作，避免文件损坏
            logger.info(f"💾 Checkpoint saved (Round {iter}, Model4_Acc {acc_list[3]:.2f}%)")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    # 保存结果
    acc_array = np.array(summary_acc_test_collect)  # Shape: (num_rounds, 4)

    # 保存为 numpy 数组（方便分析）
    np.save('shfl_acc_all_models.npy', acc_array)
    logger.info(f"💾 Saved all models accuracy to shfl_acc_all_models.npy (shape: {acc_array.shape})")

    # 分别保存每个模型的结果
    for model_idx in range(4):
        model_acc_list = acc_array[:, model_idx].tolist()
        save_result(model_acc_list, f'shfl_acc_model{model_idx+1}', args)

    # 保存最深模型（Model-4）作为主结果（向后兼容）
    save_result(acc_array[:, 3].tolist(), 'shfl_acc', args)
