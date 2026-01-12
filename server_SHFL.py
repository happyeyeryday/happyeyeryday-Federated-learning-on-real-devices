import time
import copy
import torch
from torch import nn
import numpy as np
import random
from loguru import logger
from torch.utils.data import DataLoader

# 引入工具类
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from utils.power_manager import wake_clients

# [新增] 引入 SHFL 智能体
from utils.shfl_agent import SHFLAgentManager

from models.SHFL_resnet import shfl_resnet18

# 全局变量
net_glob = None
w_local_client = []

# =============================================================================
# 1. SHFL 专用分发函数 (适配 BYOT 结构: mainblocks, bottlenecks, fcs)
# =============================================================================
def shfl_distribute(global_model_state, model_idx):
    """
    根据 model_idx (1-4) 对 BYOT 模型进行切片下发。
    
    Args:
        model_idx: 1 (Block0), 2 (Block0+1), 3 (Block0-2), 4 (Full)
    
    Structure to Keep:
    - conv1, bn1: Always
    - mainblocks.i: if i <= target_idx
    - bottlenecks.i, fcs.i: if i <= target_idx (Client需要计算中间Loss)
    """
    if model_idx == 4:
        # Full Model: 发送所有参数 (除了 BN stats)
        # 这里还是走一遍过滤逻辑比较安全
        pass
    
    local_state = {}
    # model_idx 1 -> index 0 (Block 0)
    target_exit_idx = model_idx - 1 
    
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
            
        # 3. 瓶颈层与分类头 (bottlenecks, fcs)
        elif k.startswith('bottlenecks') or k.startswith('fcs'):
            # 格式: bottlenecks.0.xxx
            try:
                parts = k.split('.')
                exit_id = int(parts[1])
                # 发送所有 <= target 的出口 (因为 Client 会计算中间 Loss)
                if exit_id <= target_exit_idx:
                    local_state[k] = v.clone()
            except: pass
            
    return local_state

# =============================================================================
# 2. SHFL 聚合函数
# =============================================================================
def shfl_aggregate(w_local_list, model_indices, net_glob):
    """
    聚合不同深度的模型 (BYOT结构)。
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
        # 这里的 model_indices[idx] 可以用来做校验，但主要依靠 Key 匹配
        for k, v_local in local_w.items():
            if k not in sum_buffer: continue
            
            # 直接累加 (不需要切片，因为 SHFL 是深度剪枝，参数维度没变，只是有的层有，有的层没有)
            sum_buffer[k] += v_local
            count_buffer[k] += 1

    # 平均并更新
    updated_state = {}
    for k in global_state.keys():
        if 'num_batches_tracked' in k:
            updated_state[k] = sum_buffer.get(k, global_state[k]) # 保持原样
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
    
    # 强制所有 BN 层进入追踪模式
    for module in test_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.track_running_stats:
                module.reset_running_stats()
            module.track_running_stats = True
            module.training = True 

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    with torch.no_grad():
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
    net.eval()
    dtLoader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)
    metric = Accumulator(2)
    with torch.no_grad():
        for images, labels in dtLoader:
            images, labels = images.to(device), labels.to(device)
            # BYOT Forward: output(list), features, embedding
            # 我们取 output 列表中的最后一个 (Full Model Prediction)
            outputs = net(images) 
            
            # 兼容性处理
            if isinstance(outputs, (tuple, list)):
                # outputs[0] 是 logits 列表
                # outputs[0][-1] 是最深层的 logits
                logits_list = outputs[0] if isinstance(outputs[0], (list, tuple)) else outputs
                final_pred = logits_list[-1]
            else:
                final_pred = outputs

            metric.add(accuracy(final_pred, labels), labels.numel())
    return metric[0] / metric[1]

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
        0: {'mac': "48:b0:2d:3c:cc:ff", 'ip': "192.168.31.154"}, # Nano
        1: {'mac': "48:b0:2d:3c:ca:18", 'ip': "192.168.31.239"}, # Nano
        2: {'mac': "48:b0:2d:3c:cc:f3", 'ip': "192.168.31.142"}, # Nano
        3: {'mac': "48:b0:2d:3c:cc:df", 'ip': "192.168.31.121"}, # Nano
        4: {'mac': "48:b0:2d:3c:ca:27", 'ip': "192.168.31.237"}, # Nano
        5: {'mac': "48:b0:2d:3c:ca:20", 'ip': "192.168.31.231"}, # Nano
        6: {'mac': "48:b0:2d:3c:ca:2e", 'ip': "192.168.31.244"}, # Nano
        7: {'mac': "48:b0:2d:2f:eb:0e", 'ip': "192.168.31.198"}, # Xavier
        8: {'mac': "48:b0:2d:2f:e2:d9", 'ip': "192.168.31.243"}, # Xavier
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

    # ================= [训练循环] =================
    for iter in range(args.epochs):
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
        wake_clients(mac_to_ip_to_wake, total_timeout=15)

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
            
            logger.info(f"📤 Send to Client {idx} (Model-{model_idx})")
            
            if connectHandler.sendData(idx, msg):
                successful_indices.append(idx)
            else:
                logger.error(f"❌ Failed to send to Client {idx}. Removing.")
                if idx in active_clients: active_clients.remove(idx)

        # 6. 接收
        w_local_client = []
        model_indices_this_round = []
        
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
            
            if 'battery_level' in msg:
                 client_states[client_idx]['E'] = msg['battery_level']
            
            if msg['type'] == 'status' and msg.get('status') == 'low_battery':
                logger.warning(f"🪫 Client {client_idx} Low Battery.")
                connectHandler.sendData(client_idx, {'type': 'shutdown_ack'})
                if client_idx in active_clients:
                    active_clients.remove(client_idx)
                    client_states[client_idx]['E'] = 0.0

            elif msg['type'] == 'net':
                logger.info(f"📥 Received from Client {client_idx}")
                w_local_client.append(msg['net'])
                model_indices_this_round.append(client_model_map[client_idx])
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
        
        acc = summary_evaluate(net_glob, dataset_test, args.device) * 100
        summary_acc_test_collect.append(acc)
        
        print("\n" + "="*60)
        print(f"📊 Round {iter}: Global Acc = {acc:.2f}%")
        print("="*60 + "\n")

    save_result(summary_acc_test_collect, 'shfl_acc', args)