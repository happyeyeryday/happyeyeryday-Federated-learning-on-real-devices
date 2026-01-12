import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
from pandas import DataFrame
import random
import numpy as np
import matplotlib
import time

from utils.ConnectHandler_client import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from utils.power_manager import smart_sleep
import copy
from loguru import logger
from models.hetero_model import resnet18 
# [新增] 引入 BatteryManager
from utils.power_manager import smart_sleep, BatteryManager 

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    
    # ========== [🔥 电池模拟 Step 1: 初始化] ==========
    
    # 1. 定义 Client ID 到设备类型的映射
    # 这个映射必须和 Server 端的角色分配一致
    DEVICE_TYPE_MAP = {
        0: 'orin',
        1: 'xavier',
        2: 'xavier',
        # 3 到 9 都是 Nano
    }
    # 默认是 nano
    device_type = DEVICE_TYPE_MAP.get(args.CID, 'nano')
    
    # 2. 实例化电池管理器
    battery_manager = BatteryManager(device_type=device_type)
    
    # 3. 预估下一次训练所需时间 (秒) - 用于低电量预判
    # 这是一个粗略的估计，你可以根据实际情况调整
    # 例如：(数据集大小 / batch_size) * local_ep * (每步平均时间)
    ESTIMATED_TRAIN_DURATION = 90 # 假设一次本地训练最多需要 90 秒
    
    # =======================================================
    
    ID = args.CID
    logger.info(f"Client {ID} ({device_type}) starting...")
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    
    loss_func = nn.CrossEntropyLoss()

    last_timestamp = time.time() # 初始化时间戳，用于计算第一次 idle 功耗

    try: 
        while True:
            # ========== [🔥 电池模拟 Step 2: 计算空闲功耗] ==========
            
            # 1. 计算从上次活动结束到现在的空闲时间
            idle_duration = time.time() - last_timestamp
            battery_manager.consume('idle', idle_duration)
            
            # 2. 记录接收数据前的时间
            recv_start_time = time.time()
            
            # 等待 Server 指令
            recv = connectHandler.receiveFromServer()
            
            # 计算接收数据+反序列化的时间
            recv_duration = time.time() - recv_start_time
            battery_manager.consume('communication', recv_duration) 
            
            if recv and recv['type'] == 'net':
                # ========== [🔥 电池模拟 Step 3: 低电量预判] ==========
                
                # 在开始训练前，检查电量是否足够完成下一次训练
                if not battery_manager.check_energy(threshold=50.0):
                        logger.warning(f"🪫 Client {ID} battery critical (<50J). Initiating shutdown protocol.")
                        
                        # 1. 发送退出通知
                        msg = {'type': 'status', 'status': 'low_battery'}
                        upload_start = time.time()
                        connectHandler.uploadToServer(msg)
                        
                        # [更新电量] 发送这一小段消息也要算电量
                        battery_manager.consume('communication', time.time() - upload_start)
                        
                        # 2. [新增] 等待 Server 确认 (ACK)
                        logger.info("⏳ Waiting for server confirmation to shutdown...")
                        ack = connectHandler.receiveFromServer()
                        
                        if ack and ack.get('type') == 'shutdown_ack':
                            logger.success("✅ Server acknowledged shutdown. Powering off.")
                        else:
                            logger.warning("⚠️ No ACK received or unknown message, forcing shutdown.")
                        
                        # 3. 退出脚本 (模拟自动关机)
                        os.system("sudo poweroff")
                        break 
                
                w_local_state_dict = recv['net']
                idxs_list = recv['idxs_list']
                round_num = recv["round"]
                rate = recv.get('rate', 1.0) 
                logger.info(f"Round {round_num}: Training with model rate {rate}")

                local_model = resnet18(model_rate=rate, track=False).to(args.device)
                
                try:
                    local_model.load_state_dict(w_local_state_dict)
                except Exception as e:
                    logger.error(f"Load state dict failed: {e}")
                    continue

                dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                    batch_size=args.bs, shuffle=True)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr * (args.lr_decay ** round_num),
                                            momentum=args.momentum, weight_decay=args.weight_decay)
                
                # ========== [🔥 电池模拟 Step 4: 计算训练功耗] ==========
                
                train_start_time = time.time()
                for _ in range(args.local_ep): # 确保使用 local_ep
                    for batch_idx, (images, labels) in enumerate(dtLoader):
                        images, labels = images.to(args.device), labels.to(args.device)
                        optimizer.zero_grad()
                        fx = local_model(images)
                        loss = loss_func(fx, labels)
                        loss.backward()
                        optimizer.step()
                train_duration = time.time() - train_start_time
                battery_manager.consume('train', train_duration)

                # ========================================================
                
                msg = dict()
                msg['type'] = 'net'
                msg['net'] = copy.deepcopy(local_model.state_dict())
                msg['rate'] = rate
                
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


                # ========================================================
                
                # 记录休眠前的时间戳
                last_timestamp = time.time()
                smart_sleep(server_ip=args.HOST)
                # 唤醒后，记录新的时间戳，用于计算休眠期间的功耗
                wake_up_time = time.time()
                sleep_duration = wake_up_time - last_timestamp
                battery_manager.consume('sleep', sleep_duration)
                last_timestamp = wake_up_time # 更新时间戳

    except (KeyboardInterrupt, ConnectionResetError, EOFError) as e:
        logger.warning(f"Connection or user interruption: {e}")
    except Exception as e:
        logger.critical(f"!!!!!!!!!!!!!! UNEXPECTED CRASH !!!!!!!!!!!!!!")
        logger.critical(f"Error Type: {type(e).__name__}")
        logger.critical(f"Error Message: {e}")
        
    finally:
        logger.critical("!!!!!!!!!!!!!! SCRIPT IS EXITING !!!!!!!!!!!!!!")