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
# 我们需要直接引用 ResNet 类和 BasicBlock，以便手动构建非标准深度的模型
from models.scalefl_resnet import ResNet, BasicBlock 
from models.scalefl_modelutils import KDLoss

# [新增] 引入 BatteryManager 和 休眠模块
from utils.power_manager import smart_sleep, BatteryManager

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def build_local_model(args, rate, exit_idx, global_ee_locs):
    # 1. 定义 ResNet18 的标准层结构
    full_layers_config = [2, 2, 2, 2] 
    
    # 2. 截断层结构
    local_layers = full_layers_config[:exit_idx + 1]
    
    # 3. 计算本地模型的总层数
    local_total_layers = sum(local_layers)
    
    # 4. 关键修改：Client 端的过滤逻辑
    # 我们需要保留所有 <= 本地总层数的 Exit
    # 对于 Xavier (total=6): [2, 4, 6] 都要保留！
    # 2->Block0, 4->Block1, 6->Block2(作为最终输出)
    # local_ee_locs = [loc for loc in global_ee_locs if loc <= local_total_layers]
    
    # ⚠️ 4. 关键修改：确定是否为全量模型
    # 如果 exit_idx 是最后一个 (3)，就是 Full Model
    is_full_model = (exit_idx == len(full_layers_config) - 1)
    
    # ⚠️ 5. 关键修改：计算 EE Locations
    if is_full_model:
        # 如果是全量，按原逻辑，小于总层数的都是 EE，最后的是 Linear
        local_ee_locs = [loc for loc in global_ee_locs if loc < local_total_layers]
    else:
        # 如果是截断模型，我们需要在截断点也放一个 EE Classifier
        # 因为 Server 会发送 ee_classifiers.{exit_idx}
        # 这里的逻辑是：找出所有 <= 当前层数的 EE 点
        # 例如 Exit=2, total=6. Global Locs=[2, 4, 6].
        # 我们需要 [2, 4, 6] 都在 local_ee_locs 里
        local_ee_locs = [loc for loc in global_ee_locs if loc <= local_total_layers]

    logger.info(f"Building Local Model: Rate={rate}, Exit_Idx={exit_idx}")
    logger.info(f" -> Layers: {local_layers}, EE_Locs: {local_ee_locs}, Is_Full: {is_full_model}")
    
    trs = bool(args.track_running_stats)
    
    # 实例化
    model = ResNet(
        BasicBlock, 
        layers=local_layers, 
        num_classes=args.num_classes, 
        ee_layer_locations=local_ee_locs, 
        scale=rate, 
        trs=trs,
        is_full_model=is_full_model  # 传入新参数
    )
    return model.to(args.device)

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # [关键参数补全] 如果 options.py 里没加这两个参数，这里给默认值
    if not hasattr(args, 'KD_T'): args.KD_T = 4.0   # 温度系数
    if not hasattr(args, 'KD_gamma'): args.KD_gamma = 0.5 # 蒸馏权重
    
    set_random_seed(args.seed)

    # ========== [🔥 电池模拟 Step 1: 初始化] ==========
    # 1. 定义 Client ID 到设备类型的映射
    DEVICE_TYPE_MAP = {
        9: 'orin',
        8: 'xavier',
        7: 'xavier',
        # 0-6 默认 nano
    }
    device_type = DEVICE_TYPE_MAP.get(args.CID, 'nano')
    
    # 2. 实例化电池管理器
    battery_manager = BatteryManager(device_type=device_type)

    ID = args.CID
    logger.info(f"Client {ID} ({device_type}) starting...")
    
    # 获取数据
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    # 原代码逻辑: connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    
    # 初始化 Loss
    # ScaleFL 使用 KDLoss (包含 CE 和 KL)
    criterion = KDLoss(args).to(args.device)
    
    local_model = None
    optimizer = None

    last_timestamp = time.time() # 初始化时间戳
    battery_state_synced = False

    try:
        while True:
            # ========== [状态 1: 空闲等待] ==========
            idle_duration = time.time() - last_timestamp
            if battery_state_synced:
                battery_manager.consume('idle', idle_duration)

            # ========== [状态 2: 接收数据] ==========
            recv_start_time = time.time()

            logger.info("Waiting for server command...")
            recv = connectHandler.receiveFromServer()

            # [更新电量] 计算通讯功耗
            recv_duration = time.time() - recv_start_time
            if not battery_state_synced:
                if recv and recv['type'] == 'net':
                    if 'battery_joules' in recv:
                        battery_manager.set_charge(recv['battery_joules'])
                    elif 'battery_level' in recv:
                        battery_manager.set_charge(float(recv['battery_level']) * battery_manager.total_capacity)
                battery_manager.consume('idle', idle_duration)
                battery_state_synced = True
            battery_manager.consume('communication', recv_duration) 

            if recv and recv['type'] == 'net':
                # ========================================================
                # [🔥 关键修改: 训练前检查电量]
                # ========================================================
                if not battery_manager.check_energy(threshold=50.0):
                    logger.warning(f"🪫 Client {ID} battery critical (<50J). Initiating shutdown protocol.")
                    
                    # 1. 发送退出通知
                    msg = {'type': 'status', 'status': 'low_battery'}
                    msg['battery_joules'] = battery_manager.get_charge()
                    msg['battery_level'] = battery_manager.get_ratio()
                    upload_start = time.time()
                    connectHandler.uploadToServer(msg)
                    battery_manager.consume('communication', time.time() - upload_start)
                    
                    # 2. 等待 Server 确认 (ACK)
                    logger.info("⏳ Waiting for server confirmation to shutdown...")
                    ack = connectHandler.receiveFromServer()
                    
                    if ack and ack.get('type') == 'shutdown_ack':
                        logger.success("✅ Server acknowledged shutdown. Powering off.")
                    else:
                        logger.warning("⚠️ No ACK received or unknown message, forcing shutdown.")
                    
                    # 3. 退出脚本 
                    os.system("sudo poweroff")
                    break

                # ========================================================
                # 电量充足，继续训练流程
                # ========================================================

                round_num = recv['round']
                w_local_state_dict = recv['net']
                idxs_list = recv['idxs_list']
                
                # 获取 ScaleFL 特定参数
                rate = recv.get('rate', 1.0)
                exit_idx = recv.get('exit_idx', 3) # 默认全深
                global_ee_locs = recv.get('global_ee_locs', [2, 4, 6])
                
                # 2. 实例化本地模型 (动态构建)
                local_model = build_local_model(args, rate, exit_idx, global_ee_locs)
                
                # 3. 加载参数
                try:
                    # strict=True 是对 ScaleFL 正确性的终极检验
                    local_model.load_state_dict(w_local_state_dict, strict=True)
                    logger.info(f"Round {round_num}: Model loaded successfully (Rate {rate}, Exit {exit_idx}).")
                except Exception as e:
                    logger.error(f"Load state dict failed: {e}")
                    # Debug info
                    logger.error(f"Local keys: {list(local_model.state_dict().keys())[:5]}")
                    logger.error(f"Recv keys: {list(w_local_state_dict.keys())[:5]}")
                    continue

                # 4. 训练准备
                dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                    batch_size=args.bs, shuffle=True)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), 
                                            lr=args.lr * (args.lr_decay ** round_num),
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay)

                # 5. 训练循环 (带自蒸馏)
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
                        
                        # ScaleFL Forward: 返回列表 [exit_0_out, exit_1_out, ..., final_out]
                        outputs = local_model(images)
                        
                        # [🔥 关键修复] 只有一个出口时不做自蒸馏
                        if len(outputs) == 1:
                            # Nano 等只有单个出口的设备，使用标准 CE Loss
                            loss = criterion.ce_loss(outputs[0], labels)
                        else:
                            # Xavier/Orin 等多出口设备，使用自蒸馏
                            # 最后一个输出作为 Teacher (Soft Target)
                            teacher_output = outputs[-1].detach() 
                            
                            loss = 0.0
                            # 遍历所有出口
                            for i, output in enumerate(outputs):
                                # 如果是最后一个出口(Teacher本身)，只算 CrossEntropy，不算 KL (gamma_active=False)
                                is_teacher = (i == len(outputs) - 1)
                                
                                # 调用 KDLoss
                                # loss_fn_kd(pred, target, soft_target, gamma_active)
                                # 注意：ScaleFL 论文中所有 exit 都要算 CE Loss
                                l = criterion.loss_fn_kd(output, labels, teacher_output, gamma_active=not is_teacher)
                                loss += l
                        
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

                # [更新电量]
                train_duration = time.time() - train_start_time
                battery_manager.consume('train', train_duration)

                logger.info(f"Round {round_num} finished. Total Avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

                # 6. 回传结果
                msg = dict()
                msg['type'] = 'net'
                # 必须回传 state_dict，以便 Server 的隐式聚合生效
                # 这里的 state_dict 天然只包含 Client 拥有的层
                msg['net'] = copy.deepcopy(local_model.state_dict())
                msg['rate'] = rate # 回传 rate 方便 Server 聚合
                msg['battery_joules'] = battery_manager.get_charge()
                msg['battery_level'] = battery_manager.get_ratio()
    
                # ========== [🔥 电池模拟 Step 5: 计算上传功耗] ==========
                upload_start_time = time.time()
                logger.info(f"📤 [Client {ID}] Uploading model...")
                connectHandler.uploadToServer(msg)
                
                # [🔥 新增: 等待 Server 确认接收]
                logger.info("⏳ Waiting for server confirmation (ACK)...")
                ack_msg = connectHandler.receiveFromServer()
                
                upload_duration = time.time() - upload_start_time
                battery_manager.consume('communication', upload_duration)
                
                if ack_msg and ack_msg.get('type') == 'upload_ack':
                    logger.success(f"✅ [Client {ID}] Server confirmed receipt. Preparing to sleep.")
                else:
                    logger.warning(f"⚠️ Did not receive standard ACK, but proceeding to sleep. Msg: {ack_msg}")


                # ========================================
                
                # 记录休眠前的时间戳
                last_timestamp = time.time()

                # ========== [状态 5: 休眠] ==========
                smart_sleep(server_ip=args.HOST)
                
                # 唤醒后，计算休眠功耗
                wake_up_time = time.time()
                sleep_duration = wake_up_time - last_timestamp
                battery_manager.consume('sleep', sleep_duration)
                
                # 更新时间戳，用于计算下一次 idle
                last_timestamp = wake_up_time

    except (KeyboardInterrupt, ConnectionResetError, EOFError) as e:
        logger.warning(f"Connection or user interruption: {e}")
    except Exception as e:
        logger.critical(f"!!!!!!!!!!!!!! UNEXPECTED CRASH !!!!!!!!!!!!!!")
        logger.critical(f"Error Type: {type(e).__name__}")
        logger.critical(f"Error Message: {e}")
    finally:
        logger.critical("!!!!!!!!!!!!!! SCRIPT IS EXITING !!!!!!!!!!!!!!")
