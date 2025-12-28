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

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    ID = args.CID
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    local_net = None
    dtLoader = None
    my_train_dict = None
    optimizer = None
    loss_func = nn.CrossEntropyLoss()

    try: 
        while True:
            recv = connectHandler.receiveFromServer()
            if recv['type'] == 'net':
                # local_net = recv['net']
                w_local_state_dict = recv['net']
                idxs_list = recv['idxs_list']
                round = recv["round"]
                # --- 新增步骤 2: 获取当前模型的压缩信息 ---
                # Server 在发送包里应该多加一个字段 'model_rate' 或 'level'
                # 例如: rate = 0.5 (表示宽度是原本的50%)
                rate = recv.get('rate', 1.0) 
                logger.info(f"Round {round}: Training with model rate {rate}")

                # [修改 3] 根据 rate 实例化本地模型
                # track=False 意味着启用 sBN (不追踪全局统计量)
                local_model = resnet18(model_rate=rate, track=False).to(args.device)
                
                # [修改 4] 加载参数
                try:
                    local_model.load_state_dict(w_local_state_dict)
                except Exception as e:
                    logger.error(f"Load state dict failed: {e}")
                    # 可能的错误：维度不匹配。请检查 server 端的切片逻辑是否和 resnet18 的 hidden_size 计算逻辑一致。
                    continue

                dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                    batch_size=args.bs, shuffle=True)
                local_model.train()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr * (args.lr_decay ** round),
                                            momentum=args.momentum, weight_decay=args.weight_decay)

                for batch_idx, (images, labels) in enumerate(dtLoader):
                    images, labels = images.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    # fx = local_net(images)['output']
                    fx = local_model(images)
                    # [步骤3] Masked CrossEntropy (简化版)
                    # 只有当你要处理严重的 Non-IID 数据时才必须加，IID 实验可以先不加
                    # if args.masked_loss:
                    #    mask = torch.zeros_like(fx)
                    #    mask[:, torch.unique(labels)] = 1
                    #    fx = fx.masked_fill(mask == 0, -1e9)
                    loss = loss_func(fx, labels)
                    loss.backward()
                    optimizer.step()

                msg = dict()
                msg['type'] = 'net'
                msg['net'] = copy.deepcopy(local_model.state_dict())
                msg['rate'] = rate
                logger.info(f"📤 [Client {ID}] Preparing to upload model)...")
                connectHandler.uploadToServer(msg)
                logger.success(f"✅ [Client {ID}] Upload complete. Now going to sleep.") # 确认上传已返回
                # connectHandler.uploadToServer(msg)
                # [🔥 插入休眠代码]
                # 训练并上传完后，立即睡觉，等待下一轮 Server 唤醒
                smart_sleep(server_ip=args.HOST)
    except Exception as e:
        # [🔥 新增] 捕获所有异常
        logger.critical(f"!!!!!!!!!!!!!! UNEXPECTED CRASH !!!!!!!!!!!!!!")
        logger.critical(f"Error Type: {type(e).__name__}")
        logger.critical(f"Error Message: {e}")
        # 在这里可以加入更多诊断信息
        
    finally:
        # [🔥 新增] 确保脚本退出时能留下遗言
        logger.critical("!!!!!!!!!!!!!! SCRIPT IS EXITING !!!!!!!!!!!!!!")
