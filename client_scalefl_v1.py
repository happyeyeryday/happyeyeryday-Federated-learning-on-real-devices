import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from loguru import logger
import time
import sys
import os
from tqdm import tqdm

# 引入基础组件
from utils.ConnectHandler_client import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed

# [关键修改] 引入 SHFL 自蒸馏组件
from models.scale_resnet_v2 import shfl_resnet18_distill
from models.scalefl_modelutils import KDLoss

# 引入 BatteryManager 和 休眠模块
from utils.power_manager import smart_sleep, BatteryManager

# 优化设置
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def build_local_model(args, model_idx):
    """
    根据 model_idx (1-4) 构建 SHFL 自蒸馏模型
    支持多出口输出用于自蒸馏（浅层学深层）
    """
    logger.info(f"Building SHFL Distill Model: Model-{model_idx}")
    
    # 使用自蒸馏版本的工厂函数
    # model_idx: 1 (Block0), 2 (Block0+1), 3 (Block0+1+2), 4 (Full)
    model = shfl_resnet18_distill(num_classes=args.num_classes, model_idx=model_idx)
    
    return model.to(args.device)

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # [ScaleFL 超参数补全]
    if not hasattr(args, 'KD_T'): args.KD_T = 4.0        # 蒸馏温度
    if not hasattr(args, 'KD_gamma'): args.KD_gamma = 0.5 # 蒸馏权重
    if not hasattr(args, 'local_ep'): args.local_ep = 5
    
    set_random_seed(args.seed)

    # ========== [1. 电池与设备初始化] ==========
    DEVICE_TYPE_MAP = {
        9: 'orin',
        8: 'xavier', 7: 'xavier',
        # 0-6 默认 nano
    }
    device_type = DEVICE_TYPE_MAP.get(args.CID, 'nano')
    battery_manager = BatteryManager(device_type=device_type)

    ID = args.CID
    logger.info(f"Client {ID} ({device_type}) starting SHFL Distill Protocol...")
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    
    # [关键] ScaleFL 损失函数 (支持 CE + KL)
    criterion = KDLoss(args).to(args.device)
    
    local_model = None
    optimizer = None
    last_timestamp = time.time()

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
                # [🔥 电量检查与保护]
                # ========================================================
                if not battery_manager.check_energy(threshold=50.0):
                    logger.warning(f"🪫 Client {ID} battery critical. Shutdown.")
                    msg = {'type': 'status', 'status': 'low_battery', 
                           'battery_level': battery_manager.current_charge / battery_manager.total_capacity}
                    connectHandler.uploadToServer(msg)
                    
                    # 等待 ACK，超时不候
                    connectHandler.receiveFromServer() 
                    os.system("sudo poweroff")
                    break

                # ========================================================
                # 解析 Server 指令
                # ========================================================
                round_num = recv['round']
                w_local_state_dict = recv['net']
                idxs_list = recv['idxs_list']
                
                # SHFL 特有参数：model_idx (1-4)
                model_idx = recv.get('model_idx', 4)  # 默认 Full Model
                
                # 2. 构建模型
                local_model = build_local_model(args, model_idx)
                
                # 3. 加载参数（strict=False 允许加载部分参数，因为有多出口）
                try:
                    local_model.load_state_dict(w_local_state_dict, strict=False)
                    logger.info(f"Round {round_num}: Distill-Model-{model_idx} loaded successfully.")
                except Exception as e:
                    logger.error(f"Load failed: {e}")
                    # 打印 key 帮助调试
                    logger.debug(f"Local keys head: {list(local_model.state_dict().keys())[:3]}")
                    continue

                # 4. 准备数据
                dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                    batch_size=args.bs, shuffle=True)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), 
                                            lr=args.lr * (args.lr_decay ** round_num),
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay)

                # ========== [状态 3: 训练 (含自蒸馏逻辑)] ==========
                train_start_time = time.time()
                epoch_loss = []
                
                # [🔥 关键] 判断是否启用自蒸馏
                # 只有当模型有多个出口时才蒸馏。Nano (Exit=0) 只有一个出口。
                # build_local_model 会根据 exit_idx 自动决定有没有 ee_classifiers
                # 我们在 forward 后通过 outputs 的长度来动态判断
                
                for epoch in range(args.local_ep):
                    batch_loss = []
                    pbar = tqdm(dtLoader, desc=f"Epoch {epoch+1}/{args.local_ep}", unit="batch")
                    
                    for batch_idx, (images, labels) in enumerate(pbar):
                        images, labels = images.to(args.device), labels.to(args.device)
                        optimizer.zero_grad()
                        
                        # Forward 返回列表 [exit1, exit2, ..., final]
                        outputs = local_model(images)
                        
                        # 兼容性检查：确保 outputs 是列表
                        if not isinstance(outputs, (list, tuple)):
                            outputs = [outputs]
                            
                        total_loss = 0.0
                        
                        # [🔥 逻辑分支]
                        if len(outputs) == 1:
                            # Case A: Nano (单出口)，只算 CE，无蒸馏
                            total_loss = nn.CrossEntropyLoss()(outputs[0], labels)
                        else:
                            # Case B: Orin/Xavier (多出口)，启用自蒸馏
                            # 最后一个输出作为 Teacher (Soft Target)
                            teacher_output = outputs[-1].detach()
                            
                            for i, output in enumerate(outputs):
                                # 最后一个出口是 Teacher 自己，不需要算 KL (gamma_active=False)
                                is_teacher_node = (i == len(outputs) - 1)
                                
                                # KDLoss 内部计算: (1-gamma)*CE + gamma*KL
                                l = criterion.loss_fn_kd(output, labels, teacher_output, 
                                                         gamma_active=not is_teacher_node)
                                total_loss += l

                        total_loss.backward()
                        
                        # 梯度裁剪 (保护 Nano)
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        batch_loss.append(total_loss.item())
                        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
                    
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))

                train_duration = time.time() - train_start_time
                battery_manager.consume('train', train_duration)
                logger.info(f"Round {round_num} finished. Avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

                # ========== [状态 4: 上传] ==========
                msg = dict()
                msg['type'] = 'net'
                # SHFL 回传完整 state_dict
                msg['net'] = copy.deepcopy(local_model.state_dict())
                msg['model_idx'] = model_idx
                msg['battery_level'] = battery_manager.current_charge / battery_manager.total_capacity
                
                upload_start_time = time.time()
                connectHandler.uploadToServer(msg)
                
                # 等待 ACK
                ack_msg = connectHandler.receiveFromServer()
                
                upload_duration = time.time() - upload_start_time
                battery_manager.consume('communication', upload_duration)
                
                if ack_msg and ack_msg.get('type') == 'upload_ack':
                    logger.success(f"✅ [Client {ID}] Upload success.")

                # ========================================
                last_timestamp = time.time()

                # ========== [状态 5: 休眠] ==========
                smart_sleep(server_ip=args.HOST)
                
                wake_up_time = time.time()
                battery_manager.consume('sleep', wake_up_time - last_timestamp)
                last_timestamp = wake_up_time

    except (KeyboardInterrupt, ConnectionResetError, EOFError) as e:
        logger.warning(f"Interrupt: {e}")
    except Exception as e:
        logger.critical(f"CRASH: {e}")
    finally:
        logger.critical("SCRIPT EXITING")