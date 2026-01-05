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
# [新增] 引入电源唤醒模块
from utils.power_manager import wake_clients

# 引入 ScaleFL 模型
from models.scalefl_resnet import resnet18_scalefl

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
    # 如果是完全体，直接深拷贝返回
    if rate == 1.0 and user_exit_idx == len(exit_block_map) - 1:
        # 依然需要过滤 BN 统计量，因为 Client 是 track=False
        pass
        
    local_state = {}
    max_allowed_block = exit_block_map[user_exit_idx]
    
    for k, v in global_model_state.items():
        # [新增] 过滤 BN 统计量 (sBN 模式)
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            continue

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
# 4. BN 校准 (更新为健壮版本)
# -------------------------------------------------------------------------
def stats(dataset, model, args):
    """
    终极版 BN 校准函数 (解决 NoneType 问题)
    """
    logger.info("Starting BN calibration...")
    test_model = copy.deepcopy(model)
    test_model.to(args.device)
    
    # 强制所有 BN 层进入追踪模式
    for module in test_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.track_running_stats:
                module.reset_running_stats()
            module.track_running_stats = True
            module.training = True 

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    batch_processed = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 50: break 
            images = images.to(args.device)
            # ScaleFL forward 返回的是 list，我们只需要跑通即可
            _ = test_model(images)
            batch_processed += 1
            
    test_model.eval()
    logger.info(f"✅ BN calibration finished after {batch_processed} batches.")
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
# 主程序
# -------------------------------------------------------------------------
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # ========== [设备配置] ==========
    # 1. 物理设备 MAC 表 (必须替换为真实 MAC)
    device_info_map = {
        0: {'mac': "3c:6d:66:28:57:90", 'ip': "192.168.31.19"}, # Orin
        1: {'mac': "48:b0:2d:2f:e2:d9", 'ip': "192.168.31.243"}, # Xavier
        2: {'mac': "48:b0:2d:2f:eb:0e", 'ip': "192.168.31.198"}, # Xavier
        3: {'mac': "48:b0:2d:3c:cc:df", 'ip': "192.168.31.121"}, # Nano
        4: {'mac': "48:b0:2d:3c:ca:27", 'ip': "192.168.31.237"}, # Nano
        5: {'mac': "48:b0:2d:3c:ca:20", 'ip': "192.168.31.231"}, # Nano
        6: {'mac': "48:b0:2d:3c:ca:2e", 'ip': "192.168.31.244"}, # Nano
        7: {'mac': "48:b0:2d:3c:cc:ff", 'ip': "192.168.31.154"}, # Nano
        8: {'mac': "48:b0:2d:3c:ca:18", 'ip': "192.168.31.239"}, # Nano
        9: {'mac': "48:b0:2d:3c:cc:f3", 'ip': "192.168.31.142"}, # Nano
    }
    
    num_physical_clients = len(device_info_map)
    # [新增] 活跃设备池
    active_clients = list(range(num_physical_clients))

    # 2. ScaleFL 性能分配表 (ID -> Rate, Exit_Idx)
    device_config_map = {}
    # Orin: Rate 1.0, Full Depth (Exit 3)
    device_config_map[0] = (1.0, 3)
    # Xavier: Rate 0.5, Depth 3 (Exit 2)
    device_config_map[1] = (0.5, 2)
    device_config_map[2] = (0.5, 2)
    # Nano: Rate 0.25, Depth 2 (Exit 1)
    for i in range(3, 10):
        device_config_map[i] = (0.25, 1)

    # 3. 全局模型配置
    global_model_config = {
        'ee_layer_locations': [2, 4, 6], 
        'scale': 1.0
    }
    exit_block_map = [0, 1, 2, 3] 

    # ========== [初始化] ==========
    logger.info("Initializing ScaleFL ResNet18...")
    net_glob = resnet18_scalefl(args, global_model_config)
    net_glob.apply(init_weights)
    net_glob.to(args.device)

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    summary_acc_test_collect = []
    time_collect = []

    logger.info(f"🔌 Server listening for {num_physical_clients} clients...")
    connectHandler = ConnectHandler(num_physical_clients, args.HOST, args.POST)

    # 目标采样数
    target_m = max(int(args.frac * num_physical_clients), 1)
    
    # ========== [训练循环] ==========
    for iter in range(args.epochs):
        # [Step 2.2] 检查存活
        if not active_clients:
            logger.critical("🪫 All clients have run out of battery. Stopping training.")
            break
            
        logger.info(f"🔋 Active Clients Pool: {active_clients} (Total: {len(active_clients)})")
        
        # 动态调整 m
        current_m = min(target_m, len(active_clients))
        
        # 采样
        if iter == 0:
            # 第一轮 Hack: 强制选中 Orin (如果它还活着)
            if 0 in active_clients and current_m > 0:
                idxs_users = [0]
                # 如果需要更多设备，从剩余活跃设备中随机选择
                if current_m > 1:
                    remaining = [c for c in active_clients if c != 0]
                    if len(remaining) >= current_m - 1:
                        idxs_users.extend(np.random.choice(remaining, current_m - 1, replace=False))
                    else:
                        idxs_users.extend(remaining)  # 全选剩余设备
                logger.warning(f"Round 0: Forced Client 0 in selection: {idxs_users}")
            else:
                idxs_users = np.random.choice(active_clients, current_m, replace=False)
        else:
            idxs_users = np.random.choice(active_clients, current_m, replace=False)
            
        print(f"\n{'='*60}")
        print(f"Round {iter}: Selected Clients {idxs_users}")
        print(f"{'='*60}\n")

        # 唤醒
        mac_to_ip_to_wake = {
            device_info_map[idx]['mac']: device_info_map[idx]['ip']
            for idx in idxs_users if idx in device_info_map
        }
        wake_clients(mac_to_ip_to_wake, total_timeout=15)

        # 下发
        for idx in idxs_users:
            device_id = idx
            current_rate, current_exit_idx = device_config_map[device_id]
            
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
            
            logger.info(f"📤 Send to Client {idx} (Rate {current_rate}, Exit {current_exit_idx})")
            
            # [关键] 检查发送是否成功
            if not connectHandler.sendData(device_id, msg):
                 logger.error(f"❌ Failed to send to Client {idx}. Removing from active list.")
                 if idx in active_clients:
                     active_clients.remove(idx)

        # [Step 2.2] 接收与处理消息
        w_local_client = []
        rates_this_round = []
        
        responses_received = 0
        expected_responses = len(idxs_users)
        
        while responses_received < expected_responses:
            try:
                msg, client_idx = connectHandler.receiveData()
                # [🔥 新增: 发送接收确认 (Upload ACK)]
                # 告诉 Client: "我收到了，你可以睡了"
                ack_msg = {'type': 'upload_ack'}
                connectHandler.sendData(client_idx, ack_msg)
            except Exception as e:
                logger.error(f"Error receiving data: {e}. Skipping.")
                responses_received += 1
                continue
                
            responses_received += 1
            
            # [处理低电量退出]
            if msg['type'] == 'status' and msg.get('status') == 'low_battery':
                logger.warning(f"🪫 Client {client_idx} reported LOW BATTERY and is exiting.")
                
                # 发送 ACK
                ack_msg = {'type': 'shutdown_ack'}
                logger.info(f"📤 Sending Shutdown ACK to Client {client_idx}...")
                if connectHandler.sendData(client_idx, ack_msg):
                    logger.success(f"✅ ACK sent to Client {client_idx}.")
                else:
                    logger.error(f"❌ Failed to send ACK to Client {client_idx}.")
                
                if client_idx in active_clients:
                    active_clients.remove(client_idx)
            
            # [处理正常模型]
            elif msg['type'] == 'net':
                logger.info(f"📥 Received model from Client {client_idx}")
                w_local_client.append(msg['net'])
                # ScaleFL 聚合需要知道 Rate
                # 这里假设 msg 里回传了 rate，或者查表
                rates_this_round.append(device_config_map[client_idx][0])
            
            else:
                logger.warning(f"Received unknown message from Client {client_idx}")

        # 聚合
        if len(w_local_client) > 0:
            logger.info(f"⚙️  Aggregating {len(w_local_client)} models...")
            w_glob = scalefl_aggregate(w_local_client, rates_this_round, net_glob)
            net_glob.load_state_dict(w_glob, strict=False)
        else:
            logger.warning("⚠️ No models received this round. Skipping aggregation.")

        # BN 校准
        net_glob_calibrated = stats(dataset_train, net_glob, args)
        # 只更新参数，保持 net_glob 的 track 状态
        net_glob.load_state_dict(net_glob_calibrated.state_dict(), strict=False)

        # 评估
        acc = summary_evaluate(net_glob, dataset_test, args.device) * 100
        
        current_time = time.time()
        summary_acc_test_collect.append(acc)
        time_collect.append(current_time)

        print("\n" + "="*60)
        print(f"📊 Round {iter}: Global Acc (Final Exit) = {acc:.2f}%")
        print("="*60 + "\n")

    save_result(summary_acc_test_collect, 'scalefl_acc', args)
    save_result(time_collect, 'scalefl_time', args)