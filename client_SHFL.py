import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from loguru import logger
import time
import sys
import os
from tqdm import tqdm  # [新增] 引入 tqdm

# 引入基础组件
from utils.ConnectHandler_client import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed

# [关键修改 1] 引入 ScaleFL 组件
from models.SHFL_resnet import shfl_resnet18

# [新增] 引入 BatteryManager 和 休眠模块
from utils.power_manager import smart_sleep, BatteryManager

# 解决 Jetson Orin 上的 CuDNN 问题
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def build_local_model(args, model_idx):
    """
    根据 model_idx (1-4) 构建 SHFL 子模型
    """
    logger.info(f"Building SHFL Local Model: Model-{model_idx}")
    
    # 使用工厂函数实例化特定深度的子模型
    # model_idx: 1 (Block0), 2 (Block0+1)... 4 (Full)
    model = shfl_resnet18(num_classes=args.num_classes, model_idx=model_idx)
    
    return model.to(args.device)

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # [关键参数补全]
    if not hasattr(args, 'local_ep'): args.local_ep = 5
    
    set_random_seed(args.seed)

    # ========== [🔥 电池模拟 Step 1: 初始化] ==========
    DEVICE_TYPE_MAP = {
        9: 'orin',
        8: 'xavier',
        7: 'xavier',
        # 0-6 默认 nano
    }
    device_type = DEVICE_TYPE_MAP.get(args.CID, 'nano')
    battery_manager = BatteryManager(device_type=device_type)

    ID = args.CID
    logger.info(f"Client {ID} ({device_type}) starting SHFL Protocol...")
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    
    loss_func = nn.CrossEntropyLoss()
    
    local_model = None
    optimizer = None

    last_timestamp = time.time() # 初始化时间戳

    try:
        while True:
            # ========== [状态 1: 空闲等待] ==========
            idle_duration = time.time() - last_timestamp
            battery_manager.consume('idle', idle_duration)

            # ========== [状态 2: 接收数据] ==========
            recv_start_time = time.time()

            logger.info("Waiting for server command...")
            recv = connectHandler.receiveFromServer()

            recv_duration = time.time() - recv_start_time
            battery_manager.consume('communication', recv_duration) 

            if recv and recv['type'] == 'net':
                # ========================================================
                # [🔥 训练前检查电量]
                # ========================================================
                if not battery_manager.check_energy(threshold=50.0):
                    logger.warning(f"🪫 Client {ID} battery critical (<50J). Initiating shutdown protocol.")
                    
                    # 1. 发送退出通知
                    msg = {'type': 'status', 'status': 'low_battery'}
                    # 附带当前电量信息 (虽然要退出了，但也发一下)
                    msg['battery_level'] = battery_manager.current_charge / battery_manager.total_capacity
                    
                    upload_start = time.time()
                    connectHandler.uploadToServer(msg)
                    battery_manager.consume('communication', time.time() - upload_start)
                    
                    # 2. 等待 Server 确认 (ACK)
                    logger.info("⏳ Waiting for server confirmation to shutdown...")
                    ack = connectHandler.receiveFromServer()
                    
                    if ack and ack.get('type') == 'shutdown_ack':
                        logger.success("✅ Server acknowledged shutdown. Powering off.")
                    else:
                        logger.warning("⚠️ No ACK received, forcing shutdown.")
                    
                    os.system("sudo poweroff")
                    break # 退出脚本

                # ========================================================
                # 电量充足，继续训练
                # ========================================================
                round_num = recv['round']
                w_local_state_dict = recv['net']
                idxs_list = recv['idxs_list']
                
                # 获取 SHFL 分配的模型 ID (1-4)
                model_idx = recv.get('model_idx', 4)
                
                # 2. 实例化本地模型 (SHFL)
                local_model = build_local_model(args, model_idx)
                
                # 3. 加载参数
                try:
                    # strict=True 验证 Server 切片是否完美匹配
                    local_model.load_state_dict(w_local_state_dict, strict=False)
                    logger.info(f"Round {round_num}: Model-{model_idx} loaded successfully.")
                except Exception as e:
                    logger.error(f"Load state dict failed: {e}")
                    # 打印一下 Keys 方便调试
                    logger.debug(f"Local keys head: {list(local_model.state_dict().keys())[:3]}")
                    logger.debug(f"Recv keys head: {list(w_local_state_dict.keys())[:3]}")
                    continue

                dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                    batch_size=args.bs, shuffle=True)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), 
                                            lr=args.lr * (args.lr_decay ** round_num),
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay)

                # ========== [状态 3: 训练] ==========
                train_start_time = time.time()
                epoch_loss = []

                logger.info(f"Starting training for {args.local_ep} epochs...")

                for epoch in range(args.local_ep):
                    batch_loss = []
                    
                    # [🔥 新增] 使用 tqdm 显示进度条
                    pbar = tqdm(dtLoader, desc=f"Epoch {epoch+1}/{args.local_ep}", unit="batch")
                    
                    for batch_idx, (images, labels) in enumerate(pbar):
                        images, labels = images.to(args.device), labels.to(args.device)
                        optimizer.zero_grad()
                        
                        # SHFL 普通训练 (Standard Cross Entropy)
                        # 如果需要自蒸馏，这里可以改，但论文主要是 RL 选模型
                        output = local_model(images)
                        
                        loss = loss_func(output, labels)
                        loss.backward()
                        
                        # [🔥 梯度裁剪] 防止极端non-IID导致的梯度爆炸
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        batch_loss.append(loss.item())
                        
                        # [🔥 新增] 更新进度条显示的 Loss
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    avg_epoch_loss = sum(batch_loss)/len(batch_loss)
                    epoch_loss.append(avg_epoch_loss)
                    # [🔥 新增] 每个 Epoch 结束打印汇总
                    logger.info(f"  Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}")

                train_duration = time.time() - train_start_time
                battery_manager.consume('train', train_duration)
                logger.info(f"Round {round_num} finished. Total Avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

                # ========== [状态 4: 上传数据] ==========
                msg = dict()
                msg['type'] = 'net'
                msg['net'] = copy.deepcopy(local_model.state_dict())
                msg['model_idx'] = model_idx
                # [🔥 关键] 回传剩余电量 (归一化 0-1) 供 Server RL 使用
                msg['battery_level'] = battery_manager.current_charge / battery_manager.total_capacity
                
                logger.info(f"📤 [Client {ID}] Uploading model & status...")
                upload_start_time = time.time()
                connectHandler.uploadToServer(msg)
                
                # 等待 ACK
                logger.info("⏳ Waiting for server confirmation (ACK)...")
                ack_msg = connectHandler.receiveFromServer()
                
                upload_duration = time.time() - upload_start_time
                battery_manager.consume('communication', upload_duration) 
                
                if ack_msg and ack_msg.get('type') == 'upload_ack':
                    logger.success(f"✅ [Client {ID}] Server confirmed receipt. Preparing to sleep.")
                
                # ========================================
                # 记录休眠前的时间戳
                last_timestamp = time.time()

                # ========== [状态 5: 休眠] ==========
                smart_sleep(server_ip=args.HOST)
                
                # 唤醒后，计算休眠功耗
                wake_up_time = time.time()
                sleep_duration = wake_up_time - last_timestamp
                battery_manager.consume('sleep', sleep_duration)
                
                # 更新时间戳
                last_timestamp = wake_up_time

    except (KeyboardInterrupt, ConnectionResetError, EOFError) as e:
        logger.warning(f"Connection or user interruption: {e}")
    except Exception as e:
        logger.critical(f"!!!!!!!!!!!!!! UNEXPECTED CRASH !!!!!!!!!!!!!!")
        logger.critical(f"Error Type: {type(e).__name__}")
        logger.critical(f"Error Message: {e}")
    finally:
        logger.critical("!!!!!!!!!!!!!! SCRIPT IS EXITING !!!!!!!!!!!!!!")