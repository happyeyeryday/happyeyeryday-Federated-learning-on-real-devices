import time

import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import random
import numpy as np

from utils.power_manager import wake_clients

from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from models.SplitModel import ResNet18_client_side, ResNet18_server_side, VGG16_client_side, VGG16_server_side, \
    ResNet8_client_side, ResNet8_server_side, ResNet18_entire, VGG16_entire, ResNet8_entire
from models.hetero_model import resnet18
import copy
import multiprocessing
import threading
from loguru import logger


net_glob = None
Reuse_ratio = 1
num_device = 10
w_local_client = []


def get_split_indices(global_k, global_v, rate):
    """
    计算切片索引的辅助函数。
    修复版：带 Debug 输出，优先保证分类器层（Linear）的输出维度不被切片。
    """
    # 1. 这种参数不切
    if 'num_batches_tracked' in global_k:
        return None, None
        
    dims = global_v.dim()
    full_out = global_v.size(0)
    
    # =======================================================
    # [关键修改] 第一步：先判断这层是不是最后一层分类器
    # =======================================================
    is_classifier = ('linear' in global_k) or ('fc' in global_k) or ('classifier' in global_k)
    
    # 用于 Debug 的标记
    log_note = "" 

    # 如果是分类器层，输出维度(类别数)必须保持完整！
    if is_classifier:
        split_out = full_out
        log_note = "[Classifier: Keep Full Out]"
    else:
        # 否则，按照 rate 缩放
        split_out = int(np.ceil(full_out * rate))
        log_note = f"[Rate {rate}]"
    
    # =======================================================
    # 第二步：根据维度返回切片
    # =======================================================
    
    slice_dim0 = None
    slice_dim1 = None
    
    # 1. Bias 或 BN weight (一维)
    if dims == 1:
        slice_dim0 = slice(0, split_out)
        # Debug 输出: 只在真正切片(rate < 1) 且是关键层时打印，或者全打印
        # if rate < 1.0:
        #     print(f" >> Slicing {global_k:30s} | Dim: 1 | {full_out} -> {split_out} | {log_note}")
        return slice_dim0, None
    
    # 2. 权重参数 (多维)
    full_in = global_v.size(1)
    split_in = int(np.ceil(full_in * rate))
    
    # [特殊情况] 第一层卷积输入 (RGB=3) 不切
    if dims == 4 and full_in == 3:
        split_in = full_in
        log_note += " [RGB Input Kept]"

    slice_dim0 = slice(0, split_out)
    slice_dim1 = slice(0, split_in)
    
    # Debug 输出: 打印权重矩阵的切片情况
    # if rate < 1.0:
    #      print(f" >> Slicing {global_k:30s} | Dim: {dims} | Out: {full_out}->{split_out}, In: {full_in}->{split_in} | {log_note}")

    return slice_dim0, slice_dim1


def hetero_distribute(global_model_state, rate):
    """
    下发函数：将大模型切成小模型
    [修改] 增加对 BN 统计量的过滤
    """
    if rate == 1.0:
        # 即使是 rate 1.0，也需要过滤掉 BN stats，因为 Client 是 track=False
        pass # 继续执行下面的逻辑

    local_state = {}
    
    for k, v in global_model_state.items():
        # [新增] 如果是 BN 统计量，直接跳过，不发送给 Client
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            continue
            
        slice_0, slice_1 = get_split_indices(k, v, rate)
        
        # 'num_batches_tracked' 在 get_split_indices 里已经处理了，但这里再加一道保险
        if slice_0 is None: 
            continue # 跳过所有不应该切的 buffer
            
        if v.dim() == 1: # Bias / BN weight
            local_state[k] = v[slice_0].clone()
            
        elif v.dim() == 2: # Linear [out, in]
            local_state[k] = v[slice_0, slice_1].clone()
            
        elif v.dim() == 4: # Conv2d [out, in, h, w]
            local_state[k] = v[slice_0, slice_1, :, :].clone()
            
    return local_state


def hetero_aggregate(w_local_list, rates_list, net_glob):
    """
    聚合函数：计数器法 (Counter-based Aggregation)
    """
    global_state = net_glob.state_dict()
    
    # 1. 初始化累加器和计数器 (在 GPU 上初始化以加速)
    device = next(net_glob.parameters()).device
    
    sum_buffer = {}
    count_buffer = {}
    
    for k, v in global_state.items():
        if 'num_batches_tracked' in k: 
            sum_buffer[k] = v.clone() # 统计量直接复制，不聚合
            count_buffer[k] = torch.ones_like(v) # 避免除以0
        else:
            sum_buffer[k] = torch.zeros_like(v, dtype=torch.float32)
            count_buffer[k] = torch.zeros_like(v, dtype=torch.float32)

    # 2. 遍历每个客户端的模型
    for idx, local_w in enumerate(w_local_list):
        rate = rates_list[idx]
        
        for k, v_local in local_w.items():
            if k not in sum_buffer or 'num_batches_tracked' in k:
                continue
            
            # 重新计算这个 rate 对应的切片位置
            # 这样就不需要缓存 param_idx 了，无状态！
            global_v = global_state[k]
            slice_0, slice_1 = get_split_indices(k, global_v, rate)
            
            # 累加到 Global Buffer 的对应位置 (Top-Left)
            if global_v.dim() == 1:
                sum_buffer[k][slice_0] += v_local
                count_buffer[k][slice_0] += 1
                
            elif global_v.dim() == 2:
                sum_buffer[k][slice_0, slice_1] += v_local
                count_buffer[k][slice_0, slice_1] += 1
                
            elif global_v.dim() == 4:
                sum_buffer[k][slice_0, slice_1, :, :] += v_local
                count_buffer[k][slice_0, slice_1, :, :] += 1

    # 3. 计算平均
    updated_state = {}
    for k in global_state.keys():
        if 'num_batches_tracked' in k:
            updated_state[k] = sum_buffer[k] # 保持原样或取第一个
            continue
            
        # 避免除以 0 (未被更新的部分保持原值)
        mask_updated = count_buffer[k] > 0
        
        # 复制旧参数作为底板
        updated_state[k] = global_state[k].clone()
        
        # 只更新有计数的部分
        if mask_updated.any():
            updated_state[k][mask_updated] = (sum_buffer[k][mask_updated] / count_buffer[k][mask_updated]).to(global_state[k].dtype)
            
    return updated_state

def stats(dataset, model, args):
    """
    终极版 BN 校准函数 (绝对正确版)
    """
    logger.info("Starting BN calibration...")
    
    test_model = copy.deepcopy(model)
    test_model.to(args.device)
    
    # [🔥 关键修复]
    for module in test_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 只有在 track=True 时，running_mean/var 才存在，才能 reset
            if module.track_running_stats:
                module.reset_running_stats()
            
            # 无论如何，都打开开关，并设为训练模式
            module.track_running_stats = True
            module.training = True 

    data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    
    batch_processed = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 50: break
            images = images.to(args.device)
            test_model(images)
            batch_processed += 1
    
    test_model.eval()
    logger.info(f"✅ BN calibration finished after {batch_processed} batches.")
    return test_model

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def summary_evaluate(net, dataset_test, device):
    net.eval()
    dtLoader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)
    metric = Accumulator(2)
    with torch.no_grad():
        for images, labels in dtLoader:
            images, labels = images.to(device), labels.to(device)
            # ---------forward prop-------------
            fx = net(images)
            metric.add(accuracy(fx, labels), labels.numel())
    return metric[0] / metric[1]


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # [配置] 设备信息表
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

    print("验证模型结构...")
    if 'resnet18' in args.model:
        # 使用统一的 hetero_model 工厂函数
        temp_net = resnet18()
        del temp_net

    num_physical_clients = len(device_info_map)
    
    # [新增] 活跃设备池，初始包含所有设备
    active_clients = list(range(num_physical_clients))

    # Rate 映射表
    device_rates_map = {}
    for i in range(num_physical_clients):
        if i == 0:          device_rates_map[i] = 1.0   # Orin
        elif i == 1 or i == 2:   device_rates_map[i] = 0.5   # Xavier
        else:               device_rates_map[i] = 0.25  # Nano

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    logger.info("Initializing HeteroFL Global Model...")
    net_glob = resnet18(model_rate=1.0, track=True)
    net_glob.apply(init_weights)
    net_glob.to(args.device)

    summary_acc_test_collect = []
    time_collect = []
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))

    logger.info(f"🔌 Server listening for {num_physical_clients} clients...")
    connectHandler = ConnectHandler(num_physical_clients, args.HOST, args.POST)

    # 目标采样数 (例如 5)
    target_m = max(int(args.frac * num_physical_clients), 1)

    for iter in range(args.epochs):
        # [Step 2.2: 检查是否有存活设备]
        if not active_clients:
            logger.critical("🪫 All clients have run out of battery. Stopping training.")
            break
        
        logger.info(f"🔋 Active Clients Pool: {active_clients} (Total: {len(active_clients)})")

        # [Step 2.2: 动态采样]
        # 如果活跃设备少于目标采样数，则全选
        current_m = min(target_m, len(active_clients))
        
        if iter == 0:
            # 第一轮 Hack: 强制选中 Orin (如果它还活着)
            idxs_users = [0, 5] if (0 in active_clients and 5 in active_clients) else np.random.choice(active_clients, current_m, replace=False)
            logger.warning("Round 0: Trying to force selection of Client 0.")
        else:
            # 从活跃池中采样
            idxs_users = np.random.choice(active_clients, current_m, replace=False)

        print("round:", iter, " choose client:", idxs_users)

        # 唤醒选中的设备
        mac_to_ip_to_wake = {
            device_info_map[idx]['mac']: device_info_map[idx]['ip']
            for idx in idxs_users if idx in device_info_map
        }
        wake_clients(mac_to_ip_to_wake, total_timeout=20) # 唤醒超时适当放宽

        # 下发模型
        for idx in idxs_users:
            device_id = idx # 物理 ID
            current_rate = device_rates_map[device_id]
            
            msg = dict()
            w_local_slice = hetero_distribute(net_glob.state_dict(), current_rate)
            msg['net'] = w_local_slice
            msg['rate'] = current_rate
            msg['idxs_list'] = dict_users[idx]
            msg['type'] = 'net'
            msg['round'] = iter
            
            logger.info("send net (rate={}) to client {}".format(current_rate, idx))
            connectHandler.sendData(device_id, msg)

        # [Step 2.2: 接收与处理退出消息]
        w_local_client = []
        client_rates_this_round = [] 
        
        # 我们期望收到 len(idxs_users) 个回复 (不论是模型还是退出通知)
        responses_received = 0
        expected_responses = len(idxs_users)
        
        while responses_received < expected_responses:
            try:
                msg, client_idx = connectHandler.receiveData()
            except Exception as e:
                logger.error(f"Error receiving data: {e}. Skipping this client.")
                responses_received += 1
                continue
            responses_received += 1
            
            if msg['type'] == 'status' and msg.get('status') == 'low_battery':
                logger.warning(f"🪫 Client {client_idx} reported LOW BATTERY and is exiting the pool.")
                
                # [🔥 新增: 发送 ACK 确认]
                ack_msg = {'type': 'shutdown_ack'}
                logger.info(f"📤 Sending Shutdown ACK to Client {client_idx}...")
                if connectHandler.sendData(client_idx, ack_msg):
                    logger.success(f"✅ ACK sent to Client {client_idx}.")
                else:
                    logger.error(f"❌ Failed to send ACK to Client {client_idx}.")

                # 从活跃列表中移除
                if client_idx in active_clients:
                    active_clients.remove(client_idx)
                # 注意：这里不把它的模型加入 w_local_client，因为它没训练

                
            # [处理正常模型]
            elif msg['type'] == 'net':
                logger.info(f"📥 Received model from Client {client_idx}")
                net = msg['net']
                w_local_client.append(net)
                rate = msg.get('rate', device_rates_map[client_idx])
                client_rates_this_round.append(rate)
                ack_msg = {'type': 'upload_ack'}
                connectHandler.sendData(client_idx, ack_msg)
                
            
            else:
                logger.warning(f"Received unknown message type from Client {client_idx}")

        # 聚合 (只聚合成功返回模型的 Client)
        if len(w_local_client) > 0:
            logger.info(f"⚙️  Aggregating {len(w_local_client)} models...")
            w_glob = hetero_aggregate(w_local_client, client_rates_this_round, net_glob)
            net_glob.load_state_dict(w_glob, strict=False)
        else:
            logger.warning("⚠️ No models received this round. Skipping aggregation.")

        # BN 校准与评估
        net_glob_calibrated = stats(dataset_train, net_glob, args)
        
        acc = summary_evaluate(net_glob_calibrated, dataset_test, args.device) * 100
        net_glob = net_glob_calibrated 

        current_time = time.time()
        summary_acc_test_collect.append(acc)
        time_collect.append(current_time)

        print("====================== SERVER V1==========================")
        print(' Test: Round {:3d}, Current time {:.3f}, Avg Accuracy {:.3f}'.format(iter, current_time, acc))
        print("==========================================================")

    save_result(summary_acc_test_collect, 'test_acc', args)
    save_result(time_collect, 'time', args)

