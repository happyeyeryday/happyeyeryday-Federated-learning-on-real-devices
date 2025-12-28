import time
import copy
import torch
import torch.nn as nn
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

# 引入 ScaleFL 模型
from models.scalefl_resnet import resnet18_scalefl, resnet110_4

# 全局变量
net_glob = None
w_local_client = []

# -------------------------------------------------------------------------
# 1. 辅助函数：计算切片索引 (Width Scaling)
# -------------------------------------------------------------------------
def get_split_indices(global_k, global_v, rate):
    if 'num_batches_tracked' in global_k:
        return None, None
        
    dims = global_v.dim()
    full_out = global_v.size(0)
    
    # 识别分类层 (包括中间的 Exit 和最后的 Linear)
    is_classifier_fc = ('linear' in global_k) or (('ee_classifiers' in global_k) and ('fc' in global_k))
    
    if is_classifier_fc:
        split_out = full_out
    else:
        split_out = int(np.ceil(full_out * rate))
    
    slice_dim0 = slice(0, split_out)
    
    if dims == 1: 
        return slice_dim0, None
    
    full_in = global_v.size(1)
    split_in = int(np.ceil(full_in * rate))
    
    if dims == 4 and full_in == 3:
        split_in = full_in

    slice_dim1 = slice(0, split_in)
    return slice_dim0, slice_dim1

# -------------------------------------------------------------------------
# 2. 核心函数：ScaleFL 下发
# -------------------------------------------------------------------------
def scalefl_distribute(global_model_state, rate, user_exit_idx, exit_block_map):
    if rate == 1.0 and user_exit_idx == len(exit_block_map) - 1:
        return copy.deepcopy(global_model_state)
        
    local_state = {}
    max_allowed_block = exit_block_map[user_exit_idx]
    
    for k, v in global_model_state.items():
        # --- Depth Pruning ---
        if 'layers.' in k:
            try:
                parts = k.split('.')
                layer_idx_in_name = int(parts[1]) 
                if layer_idx_in_name > max_allowed_block:
                    continue 
            except: pass 
                
        if 'ee_classifiers.' in k:
            try:
                parts = k.split('.')
                ee_idx = int(parts[1])
                if ee_idx > user_exit_idx:
                    continue
            except: pass

        is_last_exit = (user_exit_idx == len(exit_block_map) - 1)
        if 'linear' in k and not is_last_exit:
            continue

        # --- Width Slicing ---
        slice_0, slice_1 = get_split_indices(k, v, rate)
        
        if slice_0 is None:
            local_state[k] = v.clone()
            continue
            
        if v.dim() == 1:
            local_state[k] = v[slice_0].clone()
        elif v.dim() == 2:
            local_state[k] = v[slice_0, slice_1].clone()
        elif v.dim() == 4:
            local_state[k] = v[slice_0, slice_1, :, :].clone()
            
    return local_state

# -------------------------------------------------------------------------
# 3. 聚合函数
# -------------------------------------------------------------------------
def scalefl_aggregate(w_local_list, rates_list, net_glob):
    global_state = net_glob.state_dict()
    sum_buffer = {}
    count_buffer = {}
    
    for k, v in global_state.items():
        if 'num_batches_tracked' in k: 
            sum_buffer[k] = v.clone()
            count_buffer[k] = torch.ones_like(v)
        else:
            sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
            count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    for idx, local_w in enumerate(w_local_list):
        rate = rates_list[idx]
        for k, v_local in local_w.items():
            if k not in sum_buffer: continue 
            if 'num_batches_tracked' in k: continue

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
        if 'num_batches_tracked' in k:
            updated_state[k] = sum_buffer[k]
            continue
        mask_updated = count_buffer[k] > 0
        updated_state[k] = global_state[k].clone()
        if mask_updated.any():
            updated_state[k][mask_updated] = (sum_buffer[k][mask_updated] / count_buffer[k][mask_updated]).to(global_state[k].dtype)
            
    return updated_state

# -------------------------------------------------------------------------
# 4. BN 校准 & 评估
# -------------------------------------------------------------------------
def stats(dataset, model, args):
    print("BN Calibration...")
    test_model = copy.deepcopy(model)
    test_model.to(args.device)
    test_model.train()
    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 50: break 
            images = images.to(args.device)
            _ = test_model(images)
    test_model.eval()
    return test_model

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def summary_evaluate(net, dataset_test, device):
    net.eval()
    dtLoader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)
    metric = Accumulator(2)
    with torch.no_grad():
        for images, labels in dtLoader:
            images, labels = images.to(device), labels.to(device)
            preds_list = net(images)
            final_pred = preds_list[-1]
            metric.add(accuracy(final_pred, labels), labels.numel())
    return metric[0] / metric[1]

# -------------------------------------------------------------------------
# 主程序 (已整合你的定制配置)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # ========== [ScaleFL 真实设备配置 - 4台机器3种规格] ==========
    
    # 1. 全局模型配置
    global_model_config = {
        'ee_layer_locations': [2, 4, 6], 
        'scale': 1.0
    }
    # Exit 0 -> Block 0; Exit 1 -> Block 1; Exit 2 -> Block 2; Exit 3 -> Full
    exit_block_map = [0, 1, 2, 3]
    
    # 2. 定义 3 种设备类型的性能配置
    device_config_map = {
        0: (1.0, 3),   # Orin:   Rate 1.0,  Full Depth
        1: (0.5, 2),   # Xavier: Rate 0.5,  Exit 2 (Block 0,1,2)
        2: (0.25, 1),  # Nano:   Rate 0.25, Exit 1 (Block 0,1)
    }
    
    # 3. 定义 4 台物理机器与设备类型的映射
    # Client ID -> Device Type
    num_physical_clients = 4
    device_id_to_type_map = {
        0: 0,  # Client 0 -> Orin
        1: 1,  # Client 1 -> Xavier
        2: 2,  # Client 2 -> Nano
        3: 2,  # Client 3 -> Nano
    }
    
    # 4. 采样
    m = max(int(args.frac * num_physical_clients), 1)
    
    # ========== [初始化] ==========
    logger.info("Initializing ScaleFL ResNet18...")
    net_glob = resnet18_scalefl(args, global_model_config)
    net_glob.apply(init_weights)
    net_glob.to(args.device)
    
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    summary_acc_test_collect = []
    time_collect = []
    
    # ========== [建立连接] ==========
    logger.info(f"🔌 Server listening for {num_physical_clients} clients...")
    connectHandler = ConnectHandler(num_physical_clients, args.HOST, args.POST)
    logger.info("✅ All clients connected!")
    
    # ========== [训练循环] ==========
    for iter in range(args.epochs):
        idxs_users = np.random.choice(range(num_physical_clients), m, replace=False)
        
        print(f"\n{'='*60}")
        print(f"Round {iter}: Selected Clients {idxs_users}")
        type_names = {0: "Orin", 1: "Xavier", 2: "Nano"}
        device_types_this_round = [device_id_to_type_map[idx] for idx in idxs_users]
        print(f"Device Types: {[type_names[dt] for dt in device_types_this_round]}")
        print(f"{'='*60}\n")

        # --- [Downlink] ---
        for idx in idxs_users:
            device_type = device_id_to_type_map[idx]
            current_rate, current_exit_idx = device_config_map[device_type]
            
            msg = dict()
            w_local_slice = scalefl_distribute(
                net_glob.state_dict(), 
                rate=current_rate, 
                user_exit_idx=current_exit_idx,
                exit_block_map=exit_block_map
            )
            
            msg['net'] = w_local_slice
            msg['rate'] = current_rate
            msg['exit_idx'] = current_exit_idx
            msg['global_ee_locs'] = global_model_config['ee_layer_locations']
            msg['idxs_list'] = dict_users[idx]
            msg['type'] = 'net'
            msg['round'] = iter
            
            logger.info(f"📤 Send to Client {idx} ({type_names[device_type]}): Rate={current_rate}, Exit={current_exit_idx}")
            connectHandler.sendData(idx, msg)

        # --- [Uplink] ---
        w_local_client = []
        rates_this_round = []
        
        while len(w_local_client) < len(idxs_users):
            msg, client_idx = connectHandler.receiveData()
            if msg['type'] == 'net':
                w_local_client.append(msg['net'])
                rates_this_round.append(msg['rate'])
                logger.info(f"📥 Received from Client {client_idx}")

        # --- [Aggregation] ---
        logger.info("⚙️  Aggregating...")
        w_glob = scalefl_aggregate(w_local_client, rates_this_round, net_glob)
        net_glob.load_state_dict(w_glob)

        # --- [BN Calibration] ---
        # logger.info("🔧 BN Calibration...")
        net_glob_calibrated = stats(dataset_train, net_glob, args)
        net_glob = net_glob_calibrated

        # --- [Evaluate] ---
        acc = summary_evaluate(net_glob, dataset_test, args.device) * 100
        
        current_time = time.time()
        summary_acc_test_collect.append(acc)
        time_collect.append(current_time)

        print("\n" + "="*60)
        print(f"📊 Round {iter}: Global Acc (Final Exit) = {acc:.2f}%")
        print("="*60 + "\n")

    save_result(summary_acc_test_collect, 'scalefl_acc', args)
    save_result(time_collect, 'scalefl_time', args)