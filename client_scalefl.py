import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from loguru import logger

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
    
    # 获取数据
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    ID = args.CID
    # 假设 num_device_types = 4 (对应 Server 端的设定)
    # Client ID 决定了其物理身份，连接时使用 ID % num_types ? 
    # 原代码逻辑: connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    # 我们保持原样，让 ConnectHandler 处理连接细节
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    
    # 初始化 Loss
    # ScaleFL 使用 KDLoss (包含 CE 和 KL)
    criterion = KDLoss(args).to(args.device)
    
    local_model = None
    optimizer = None

    while True:
        # 1. 接收 Server 消息
        recv = connectHandler.receiveFromServer()
        if recv['type'] == 'net':
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
                # 如果 build_local_model 构建的结构和 server 发来的 state_dict 完全匹配
                # 这里就不会报错。如果报错，说明“隐式过滤”或“模型构建”有 Bug。
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
            epoch_loss = []
            for epoch in range(args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(dtLoader):
                    images, labels = images.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    
                    # ScaleFL Forward: 返回列表 [exit_0_out, exit_1_out, ..., final_out]
                    outputs = local_model(images)
                    
                    # 计算 Loss
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
                    
                    # 也可以选择加权求和，ScaleFL 论文通常是直接 sum
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            logger.info(f"Round {round_num} finished. Avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

            # 6. 回传结果
            msg = dict()
            msg['type'] = 'net'
            # 必须回传 state_dict，以便 Server 的隐式聚合生效
            # 这里的 state_dict 天然只包含 Client 拥有的层
            msg['net'] = copy.deepcopy(local_model.state_dict())
            msg['rate'] = rate # 回传 rate 方便 Server 聚合
            
            connectHandler.uploadToServer(msg)