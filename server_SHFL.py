import time
import copy
import torch
import numpy as np
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

# 引入 SHFL 模型 (基于 BYOT)
from models.SHFL_resnet import shfl_resnet18

# 全局变量
net_glob = None

# =============================================================================
# 1. SHFL 专用分发函数 (适配 BYOT 结构)
# =============================================================================
def shfl_distribute(global_model_state, model_idx):
    """
    根据 model_idx (1-4) 对 BYOT 模型进行切片下发。
    
    Model Structure:
    - mainblocks.0, .1, .2, .3
    - bottlenecks.0, .1, .2, .3
    - fcs.0, .1, .2, .3
    
    Logic:
    - If model_idx = 2 (Block 0+1):
        - Send: mainblocks.0, mainblocks.1
        - Send: bottlenecks.1 (对应的出口)
        - Send: fcs.1 (对应的分类头)
        - Send: conv1, bn1 (公共头部)
    """
    if model_idx == 4:
        # Full Model: 发送所有参数
        return copy.deepcopy(global_model_state)
    
    local_state = {}
    target_exit_idx = model_idx - 1 # model_idx 1 -> index 0
    
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
                block_id = int(k.split('.')[1])
                # 只发送 <= target_exit_idx 的块
                # 例如 Model-2 (idx=1): 需要 Block 0, 1
                if block_id <= target_exit_idx:
                    local_state[k] = v.clone()
            except: pass
            
        # 3. 瓶颈层与分类头 (bottlenecks, fcs)
        # SHFL 论文通常只训练对应深度的那个出口，或者训练所有浅层出口
        # 为了简化且符合 BYOT 逻辑，我们只下发**当前深度对应的那个出口**
        elif k.startswith('bottlenecks') or k.startswith('fcs'):
            try:
                exit_id = int(k.split('.')[1])
                # 只发送当前出口
                if exit_id == target_exit_idx:
                    local_state[k] = v.clone()
            except: pass
            
    return local_state

# =============================================================================
# 2. SHFL 聚合函数
# =============================================================================
def shfl_aggregate(w_local_list, model_indices, net_glob):
    """
    聚合不同深度的模型。
    model_indices: list, 每个 Client 对应的 model_idx (1-4)
    """
    global_state = net_glob.state_dict()
    sum_buffer = {}
    count_buffer = {}
    
    # 初始化
    for k, v in global_state.items():
        if 'num_batches_tracked' in k: continue
        sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
        count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    for idx, local_w in enumerate(w_local_list):
        # 这里的 model_idx 暂时没用到，因为我们依靠 key 匹配
        # 只有 Client 拥有这个 key，它才会传回来
        
        for k, v_local in local_w.items():
            if k not in sum_buffer: continue
            
            # SHFL 不涉及宽度切片，直接全量聚合
            sum_buffer[k] += v_local
            count_buffer[k] += 1

    # 平均
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
# 3. 工具函数
# =============================================================================
def stats(dataset, model, args):
    logger.info("Starting BN calibration...")
    test_model = copy.deepcopy(model)
    test_model.to(args.device)
    
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
            # BYOT 模型返回 (output, features...)，我们不需要返回值
            _ = test_model(images)
    
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
            # BYOT Forward 返回: output(list), features, embedding
            # 对应的 output 是 [logits_1, logits_2, logits_3, logits_4]
            # SHFL 评估通常看最深层出口 (Model-4)
            # 注意：如果网络结构被裁剪了，这里可能会报错，但在 Server 端我们用的是完整的 net_glob
            outputs = net(images) 
            # 兼容性处理：如果返回 tuple/list
            if isinstance(outputs, (tuple, list)):
                final_pred = outputs[-1] # 取最后一个出口的 logits
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

    # 1. 设备配置 (MAC & IP)
    device_info_map = {
        0: {'mac': "3c:6d:66:28:57:90", 'ip': "192.168.31.19", 'type': 'orin'}, 
        1: {'mac': "48:b0:2d:2f:e2:d9", 'ip': "192.168.31.243", 'type': 'xavier'}, 
        2: {'mac': "48:b0:2d:2f:eb:0e", 'ip': "192.168.31.198", 'type': 'xavier'}, 
        3: {'mac': "48:b0:2d:3c:cc:df", 'ip': "192.168.31.121", 'type': 'nano'}, 
        4: {'mac': "48:b0:2d:3c:ca:27", 'ip': "192.168.31.237", 'type': 'nano'}, 
        5: {'mac': "48:b0:2d:3c:ca:20", 'ip': "192.168.31.231", 'type': 'nano'}, 
        6: {'mac': "48:b0:2d:3c:ca:2e", 'ip': "192.168.31.244", 'type': 'nano'}, 
        7: {'mac': "48:b0:2d:3c:cc:ff", 'ip': "192.168.31.154", 'type': 'nano'}, 
        8: {'mac': "48:b0:2d:3c:ca:18", 'ip': "192.168.31.239", 'type': 'nano'}, 
        9: {'mac': "48:b0:2d:3c:cc:f3", 'ip': "192.168.31.142", 'type': 'nano'}, 
    }
    
    # 2. 硬件能力配置 (用于 RL State: [L, C, E])
    # 归一化常数
    MAX_COMPUTE = 1.0 # Orin=1.0
    MAX_ENERGY = 100000.0 # Orin Max J
    
    # 设备算力表 (C)
    COMPUTE_CAPABILITY = {
        'orin': 1.0,
        'xavier': 0.5,
        'nano': 0.25
    }
    
    num_physical_clients = len(device_info_map)
    active_clients = list(range(num_physical_clients))
    
    # 3. 维护每个 Client 的当前状态 (L, C, E)
    # 初始化电量满电
    client_states = {}
    for idx, info in device_info_map.items():
        client_states[idx] = {
            'C': COMPUTE_CAPABILITY[info['type']],
            'E': 1.0, # 归一化电量 (Current / Max)
            # L (数据量) 在 get_dataset 后填充
        }

    # ================= [初始化] =================
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    # 填充 L (数据量)
    max_data_len = max([len(dict_users[i]) for i in range(num_physical_clients)])
    for idx in client_states:
        client_states[idx]['L'] = len(dict_users[idx]) / max_data_len

    logger.info("Initializing SHFL Global Model (BYOT)...")
    net_glob = shfl_resnet18(num_classes=args.num_classes)
    net_glob.apply(init_weights)
    net_glob.to(args.device)
    
    # ================= [RL Agent 初始化] =================
    # 注意：这里的 model_dir 指向存放 .pkl 的目录
    try:
        rl_agent = SHFLAgentManager(real_n_agents=num_physical_clients, model_dir='./models/SHFL')
        logger.success("RL Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to init RL Agent: {e}")
        exit(1)

    summary_acc_test_collect = []
    time_collect = []
    
    logger.info(f"🔌 Server listening for {num_physical_clients} clients...")
    connectHandler = ConnectHandler(num_physical_clients, args.HOST, args.POST)

    # ================= [训练循环] =================
    for iter in range(args.epochs):
        if not active_clients:
            logger.critical("🪫 All clients have run out of battery.")
            break
            
        logger.info(f"🔋 Active Clients Pool: {active_clients} (Total: {len(active_clients)})")
        
        # 1. 准备 RL 输入状态
        # 需要构建一个 list, 顺序对应 Client 0, 1, 2...
        observations = []
        for i in range(num_physical_clients):
            # 即使 client 不活跃，也要占位 (RL 需要固定维度输入)
            # 不活跃的 client 电量可以设为 0
            if i in active_clients:
                observations.append(client_states[i])
            else:
                dead_state = client_states[i].copy()
                dead_state['E'] = 0.0
                observations.append(dead_state)
        
        # 2. RL 决策 (Dual-Selection)
        # actions: list of int (0=不选, 1-4=Model1-4)
        actions = rl_agent.select_models(observations, round_num=iter)
        
        # 解析选中的 Clients
        idxs_users = []
        client_model_map = {} # ID -> Model_Idx
        
        for idx, action in enumerate(actions):
            if idx in active_clients and action > 0: # action > 0 意味着被选中
                idxs_users.append(idx)
                client_model_map[idx] = action
        
        # 转换为 numpy 数组以兼容后续代码
        idxs_users = np.array(idxs_users)
        
        print(f"\n{'='*60}")
        print(f"Round {iter}: RL Selected {len(idxs_users)} Clients")
        print(f"Selection: {idxs_users}")
        print(f"Models: {[client_model_map[i] for i in idxs_users]}")
        print(f"{'='*60}\n")
        
        if len(idxs_users) == 0:
            logger.warning("RL Agent selected NO clients this round! Skipping...")
            time.sleep(2)
            continue

        # 3. 唤醒
        mac_to_ip_to_wake = {
            device_info_map[idx]['mac']: device_info_map[idx]['ip']
            for idx in idxs_users
        }
        wake_clients(mac_to_ip_to_wake, total_timeout=15)

        # 4. 下发
        successful_indices = []
        for idx in idxs_users:
            model_idx = client_model_map[idx]
            
            msg = dict()
            # 调用 SHFL 切片函数
            w_local_slice = shfl_distribute(net_glob.state_dict(), model_idx)
            
            msg['net'] = w_local_slice
            msg['model_idx'] = model_idx # 告诉 Client 跑哪个 Model (1-4)
            msg['idxs_list'] = dict_users[idx]
            msg['type'] = 'net'
            msg['round'] = iter
            
            logger.info(f"📤 Send to Client {idx} (Model-{model_idx})")
            
            if connectHandler.sendData(idx, msg):
                successful_indices.append(idx)
            else:
                logger.error(f"❌ Failed to send to Client {idx}.")
                if idx in active_clients:
                    active_clients.remove(idx)

        # 5. 接收
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
            
            # [更新电量状态]
            if 'battery_level' in msg:
                 # 假设 Client 回传的是剩余 Wh 或 J，这里再次归一化
                 # 简单起见，Client 直接回传剩余百分比 (0.0-1.0) 最方便
                 # 或者 Server 自己估算。这里假设 msg['battery_level'] 是 0-1
                 client_states[client_idx]['E'] = msg['battery_level']
            
            if msg['type'] == 'status' and msg.get('status') == 'low_battery':
                logger.warning(f"🪫 Client {client_idx} Low Battery.")
                connectHandler.sendData(client_idx, {'type': 'shutdown_ack'})
                if client_idx in active_clients:
                    active_clients.remove(client_idx)
                    client_states[client_idx]['E'] = 0.0 # 标记为死

            elif msg['type'] == 'net':
                logger.info(f"📥 Received from Client {client_idx}")
                w_local_client.append(msg['net'])
                model_indices_this_round.append(client_model_map[client_idx])
                
                # 发送 ACK (保持一致性)
                connectHandler.sendData(client_idx, {'type': 'upload_ack'})

        # 6. 聚合
        if len(w_local_client) > 0:
            logger.info(f"⚙️ Aggregating {len(w_local_client)} models...")
            w_glob = shfl_aggregate(w_local_client, model_indices_this_round, net_glob)
            net_glob.load_state_dict(w_glob, strict=False)
        
        # 7. 校准与评估
        net_glob_calibrated = stats(dataset_train, net_glob, args)
        net_glob.load_state_dict(net_glob_calibrated.state_dict(), strict=False)
        
        acc = summary_evaluate(net_glob, dataset_test, args.device) * 100
        summary_acc_test_collect.append(acc)
        
        print("\n" + "="*60)
        print(f"📊 Round {iter}: Global Acc = {acc:.2f}%")
        print("="*60 + "\n")

    save_result(summary_acc_test_collect, 'shfl_acc', args)