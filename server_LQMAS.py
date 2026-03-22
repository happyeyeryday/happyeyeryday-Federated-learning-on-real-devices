import time
import torch
from torch import nn
import numpy as np
import copy
from loguru import logger
import sys
import os
from collections import OrderedDict

# 导入原有工具
from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from models.SplitModel import ResNet18_entire, VGG16_entire, ResNet8_entire

# 导入QMIX控制器
from Qmix_controller import QMIXController

# 全局变量
net_glob = None
Reuse_ratio = 2
num_device = 0
w_local_client = []
client_resources = {}  # 存储客户端资源信息
client_quant_errors = {}  # 存储客户端量化误差

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def weighted_aggregation(client_weights, client_updates, client_info, global_model, logger):
    """精度感知的加权聚合方法：根据量化误差、验证精度等动态调整聚合权重"""
    weight_dict = OrderedDict()
    weight_difference_dict = OrderedDict()
    
    # 从客户端获取量化误差
    quant_errors = [info.get('quant_error', 0.1) for info in client_info]
    
    # 计算基于量化误差的权重: p_i^* = \frac{\frac{1}{1 + q_i}}{\sum_{i \in [n]} \frac{1}{1 + q_i}}
    inverse_error_weights = [1.0 / (1.0 + error) for error in quant_errors]
    sum_inverse_errors = sum(inverse_error_weights)
    
    # 防止除以零
    if sum_inverse_errors < 1e-10:
        error_weights = [1.0 / len(client_info)] * len(client_info)
    else:
        error_weights = [weight / sum_inverse_errors for weight in inverse_error_weights]
    
    # 获取验证精度
    val_accuracies = [info.get('validation_accuracy', 0.0) for info in client_info]
    
    # 归一化验证精度
    max_acc = max(val_accuracies) if val_accuracies else 1.0
    if max_acc > 0:
        # 使用相对精度作为权重因子
        acc_weights = [acc/max_acc for acc in val_accuracies]
    else:
        acc_weights = [1.0/len(client_info)] * len(client_info)
    
    # 综合考虑其他因素（量化精度、数据量）
    final_weights = []
    
    # 计算总数据量
    total_samples = sum([info.get('samples', 0) for info in client_info])
    
    for i, info in enumerate(client_info):
        # 1. 量化精度权重：高精度客户端获得更高权重
        quant_weight = (info.get('quant_budget', 32) / 32.0) ** 1.2
        
        # 2. 数据量权重：拥有更多数据的客户端获得更高权重
        data_weight = info.get('samples', 0) / total_samples if total_samples > 0 else 1.0
        
        # 组合权重（调整各因素的重要性）
        weight = error_weights[i] * 0.5 + acc_weights[i] * 0.3 + data_weight * 0.2
        final_weights.append(weight)
        
        # 记录日志
        logger.info(f"客户端{i}：量化={info.get('quant_budget', 32)}位, " +
                   f"数据量={info.get('samples', 0)}, " +
                   f"验证精度={val_accuracies[i]:.2f}%, " +
                   f"量化误差={quant_errors[i]:.6f}, " +
                   f"权重={weight:.4f}")
    
    # 归一化权重
    total_weight = sum(final_weights)
    if total_weight > 0:
        final_weights = [w/total_weight for w in final_weights]
    else:
        # 如果所有权重都为0（理论上不应该发生），则使用均匀权重
        final_weights = [1.0/len(client_info) for _ in client_info]
    
    # 使用计算出的权重进行加权聚合
    for i, (weight, model_diff) in enumerate(zip(final_weights, client_updates)):
        if i == 0:
            for key in model_diff.keys():
                weight_difference_dict[key] = model_diff[key] * weight
        else:
            for key in model_diff.keys():
                weight_difference_dict[key] += model_diff[key] * weight
    
    # 应用聚合后的更新到全局模型
    for key in weight_difference_dict.keys():
        weight_dict[key] = global_model.state_dict()[key] + weight_difference_dict[key]
    
    return weight_dict

def FedAvg(w):
    """原始FedAvg方法（均等权重）"""
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
            fx = net(images)['output']
            metric.add(accuracy(fx, labels), labels.numel())
    return metric[0] / metric[1]

def calculate_reward(prev_acc, current_acc, energy_consumption, total_time, successful_clients_ratio):
    """计算奖励函数"""
    # 精度提升奖励
    acc_reward = (current_acc - prev_acc) * 100  # 放大差异
    
    # 能源效率奖励
    energy_penalty = -0.01 * energy_consumption  # 能耗越低越好
    
    # 时间效率奖励
    time_penalty = -0.05 * total_time  # 用时越短越好
    
    # 成功率奖励
    success_reward = successful_clients_ratio * 5  # 成功率越高越好
    
    # 综合奖励
    reward = acc_reward * 0.6 + energy_penalty * 0.15 + time_penalty * 0.15 + success_reward * 0.1
    
    return reward

def create_validation_set(dataset_test, num_clients):
    """为每个客户端创建验证集索引"""
    val_indices = list(range(len(dataset_test)))
    np.random.shuffle(val_indices)
    
    # 为每个客户端分配相同的验证集（使用所有测试数据）
    # 也可以选择分割验证集，但共享验证集可以更好地比较客户端性能
    client_val_indices = {i: val_indices for i in range(num_clients)}
    
    logger.info(f"创建了验证集，每个客户端有 {len(val_indices)} 个验证样本")
    return client_val_indices

def broadcast_validation_set(connectHandler, client_val_indices, args):
    """向所有客户端广播验证集"""
    logger.info("开始向客户端分发验证集...")
    
    # 计算实际设备数量
    num_device = int(args.num_users / Reuse_ratio)
    
    # 为每个逻辑客户端发送验证集索引
    for logical_client_idx in range(args.num_users):
        msg = dict()
        msg['type'] = 'validation_set'
        msg['validation_indices'] = client_val_indices[logical_client_idx]
        msg['client_id'] = logical_client_idx  # 添加逻辑客户端ID
        
        # 计算物理设备索引
        physical_device_idx = logical_client_idx % num_device
        
        connectHandler.sendData(physical_device_idx, msg)
        logger.info(f"向物理设备 {physical_device_idx} 发送逻辑客户端 {logical_client_idx} 的验证集索引")
    
    # 等待所有客户端确认
    confirmation_count = 0
    client_resources = {}
    
    while confirmation_count < args.num_users:
        msg, physical_device_idx = connectHandler.receiveData()
        
        if msg['type'] == 'validation_confirmed':
            confirmation_count += 1
            # 获取逻辑客户端ID
            logical_client_idx = msg.get('client_id', physical_device_idx)
            
            # 存储客户端资源信息
            if 'resource' in msg:
                client_resources[logical_client_idx] = msg['resource']
                logger.info(f"收到逻辑客户端 {logical_client_idx} (物理设备 {physical_device_idx}) 确认，" + 
                          f"电量: {msg['resource']['battery_level']:.1f}J")
    
    logger.info(f"所有 {confirmation_count} 个逻辑客户端已确认接收验证集")
    return client_resources


def broadcast_model_for_validation(connectHandler, model, dict_users, round_idx, args):
    """向所有客户端广播模型进行验证"""
    logger.info(f"阶段1: 向所有客户端广播模型进行验证")
    
    # 计算实际设备数量
    num_device = int(args.num_users / Reuse_ratio)
    
    for logical_client_idx in range(args.num_users):
        msg = dict()
        msg['type'] = 'validation'
        msg['net'] = copy.deepcopy(model)
        msg['idxs_list'] = dict_users[logical_client_idx]
        msg['round'] = round_idx
        msg['client_id'] = logical_client_idx  # 添加逻辑客户端ID
        
        # 计算物理设备索引
        physical_device_idx = logical_client_idx % num_device
        
        connectHandler.sendData(physical_device_idx, msg)
        logger.info(f"向物理设备 {physical_device_idx} 发送逻辑客户端 {logical_client_idx} 的验证请求")
    
    # 等待所有客户端验证结果
    validation_results = {}
    received_count = 0
    
    while received_count < args.num_users:
        msg, physical_device_idx = connectHandler.receiveData()
        
        if msg['type'] == 'validation_result':
            received_count += 1
            # 获取逻辑客户端ID
            logical_client_idx = msg.get('client_id', physical_device_idx)
            
            validation_results[logical_client_idx] = {
                'accuracy': msg.get('accuracy', 0.0),
                'resource': msg.get('resource', {}),
                'training_time': msg.get('training_time', 0.0),
                'energy': msg.get('energy', 0.0)
            }
            
            # 更新客户端资源信息
            if 'resource' in msg:
                client_resources[logical_client_idx] = msg['resource']
            
            logger.info(f"收到逻辑客户端 {logical_client_idx} (物理设备 {physical_device_idx}) 验证结果: " + 
                      f"准确率 {msg.get('accuracy', 0.0):.2f}%, 电量 {msg.get('resource', {}).get('battery_level', 0.0):.1f}J")
    
    logger.info(f"收到所有 {received_count} 个逻辑客户端的验证结果")
    return validation_results

def select_clients_with_qmix(qmix_controller, client_resources, dict_users, args, global_state, validation_results, exploration_rate):
    """使用QMIX选择客户端和分配量化精度"""
    logger.info(f"阶段2: 使用QMIX选择客户端和分配量化精度")
    
    # 获取每个客户端的观察向量
    observations = []
    for i in range(args.num_users):
        # 默认值
        obs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 六维观察向量
        
        # 如果有验证结果则使用实际值
        client_idx = i % (args.num_users // Reuse_ratio)
        if client_idx in validation_results:
            res = validation_results[client_idx]['resource']
            obs[0] = res.get('battery_level', 50.0) / 100.0  # 归一化电量
            obs[1] = res.get('computing_power', 5.0) / 10.0  # 归一化计算能力
            obs[2] = res.get('bandwidth', 5.0) / 10.0  # 归一化带宽
            obs[3] = len(dict_users[i]) / 1000.0  # 归一化数据大小
            obs[4] = validation_results[client_idx].get('accuracy', 50.0) / 100.0  # 归一化验证精度
            obs[5] = validation_results[client_idx].get('training_time', 30.0) / 60.0  # 归一化训练时间
        
        observations.append(obs)
    
    # 记录QMIX决策信息
    logger.info("----- QMIX决策信息 -----")
    logger.info(f"当前全局精度: {global_state[0]:.4f}, 探索率: {exploration_rate:.4f}")
    
    # 使用QMIX选择动作
    qmix_controller.reset_hidden_states()
    actions = qmix_controller.select_actions(observations, global_state, epsilon=exploration_rate)
    
    # 显示客户端状态和QMIX决策
    logger.info("客户端状态和QMIX决策:")
    for i, (obs, action) in enumerate(zip(observations, actions)):
        if action > 0:  # action=0表示不参与
            quant_budget = args.quant_budget_options[action-1]
            logger.info(f"客户端 {i}: 电量={obs[0]*100:.1f}%, 计算能力={obs[1]*10:.2f}, " +
                       f"数据量={obs[3]*1000:.0f}, 验证精度={obs[4]*100:.2f}%, 选择量化精度={quant_budget}位")
    
    # 计算每个客户端的价值并检查电量状态
    client_values = []
    available_clients = []
    
    for i, action in enumerate(actions):
        if action > 0:  # action=0表示不参与
            # 检查电量是否足够
            client_idx = i % (args.num_users // Reuse_ratio)
            if client_idx in client_resources:
                battery_level = client_resources[client_idx].get('battery_level', 0.0)
                
                # 估计所需能量 (简化计算)
                training_power = client_resources[client_idx].get('training_power', 10.0)
                required_energy = training_power * args.local_ep * 5.0  # 假设每个epoch 5秒
                
                if battery_level > required_energy + 5.0:  # 电量阈值 (保留5%安全余量)
                    # 电量充足，计算价值
                    quant_budget = args.quant_budget_options[action-1]
                    
                    # 价值计算综合考虑: 量化精度、功率、电量水平
                    quant_factor = quant_budget / 16.0  
                    power_factor = 15.0 / max(1.0, training_power)
                    battery_factor = min(1.0, battery_level / 500.0) ** 0.5  # 电量因子
                    
                    value = quant_factor * power_factor * battery_factor
                    client_values.append(value)
                    available_clients.append(i)
                else:
                    # 电量不足
                    client_values.append(-1)  # 负值确保不会被选中
                    logger.info(f"客户端 {i}: 电量不足 ({battery_level:.1f}% / 需要: {required_energy:.1f}%), 不可用")
            else:
                client_values.append(-1)
    
    # 检查可用客户端数量
    if len(available_clients) == 0:
        logger.warning("没有可用的客户端！所有客户端电量都不足。")
        return [], {}
    
    # 选择客户端
    m = max(int(args.frac * args.num_users), 1)
    
    if len(available_clients) < m:
        logger.warning(f"可用客户端数量({len(available_clients)})少于目标数量({m})!")
        selected_clients = available_clients
    else:
        # 从可用客户端中选择价值最高的
        sorted_indices = np.argsort(client_values)
        selected_indices = sorted_indices[-m:]
        selected_clients = [idx for idx in selected_indices if client_values[idx] >= 0]  # 过滤掉不可用客户端
    
    # 分配量化精度
    client_quant_budgets = {}
    for idx in selected_clients:
        action_idx = actions[idx] - 1  # 动作索引转换为量化预算索引
        client_quant_budgets[idx] = args.quant_budget_options[max(0, min(action_idx, len(args.quant_budget_options)-1))]
    
    logger.info(f"选择的客户端: {selected_clients}")
    logger.info(f"分配的量化精度: {[client_quant_budgets.get(idx, 16) for idx in selected_clients]}")
    logger.info("------------------------")
    
    return selected_clients, client_quant_budgets, observations, actions

def train_selected_clients(connectHandler, model, dict_users, selected_clients, client_quant_budgets, round_idx, args):
    """训练选定的客户端 (已修复死锁BUG的健壮版本)"""
    logger.info(f"阶段3: 训练选定的客户端")
    
    if not selected_clients:
        logger.warning("没有选定的客户端，跳过本轮训练。")
        return [], []

    # 计算实际设备数量
    num_device = int(args.num_users / Reuse_ratio)

    # 按物理设备分组逻辑客户端
    device_clients = {}
    for logical_client_idx in selected_clients:
        physical_device_idx = logical_client_idx % num_device
        if physical_device_idx not in device_clients:
            device_clients[physical_device_idx] = []
        device_clients[physical_device_idx].append(logical_client_idx)
    
    # 向每个物理设备发送所有分配给它的逻辑客户端训练任务
    for physical_device_idx, logical_clients in device_clients.items():
        logger.info(f"向物理设备 {physical_device_idx} 发送 {len(logical_clients)} 个逻辑客户端的训练任务")
    
        # 发送所有训练任务
        for logical_client_idx in logical_clients:
            quant_budget = client_quant_budgets.get(logical_client_idx, 32)
            
            msg = dict()
            msg['type'] = 'train'
            msg['net'] = copy.deepcopy(model)
            msg['idxs_list'] = dict_users[logical_client_idx]
            msg['round'] = round_idx
            msg['quant_budget'] = quant_budget
            msg['client_id'] = logical_client_idx
            
            logger.info(f"向物理设备 {physical_device_idx} 发送逻辑客户端 {logical_client_idx} 的训练请求，量化精度: {quant_budget}位")
            connectHandler.sendData(physical_device_idx, msg)


        # 发送批量处理开始信号
        batch_msg = dict()
        batch_msg['type'] = 'process_pending_trains'
        batch_msg['expected_results'] = len(logical_clients)
        batch_msg['logical_clients'] = logical_clients
        
        logger.info(f"向物理设备 {physical_device_idx} 发送批量处理开始信号")
        connectHandler.sendData(physical_device_idx, batch_msg)
    

    # 2. 智能地接收客户端返回的模型更新 (关键修复)
    client_updates = []
    client_info = []
    
    # 创建一个集合来追踪哪些被选中的客户端还未返回结果
    waiting_for_clients = set(selected_clients)
    
    timeout = 18000  # 30分钟的超时时间
    start_wait = time.time()
    
    logger.info(f"等待 {len(waiting_for_clients)} 个客户端返回训练结果: {sorted(list(waiting_for_clients))}")

    while len(waiting_for_clients) > 0:
        # 检查是否超时
        if time.time() - start_wait > timeout:
            logger.warning(f"等待超时！仍有 {len(waiting_for_clients)} 个客户端未响应: {sorted(list(waiting_for_clients))}")
            break
        
        # 接收任何到达的消息
        msg, physical_device_idx = connectHandler.receiveData()
        
        # 检查消息类型是否正确
        if msg.get('type') != 'train_result':
            logger.warning(f"收到非预期的消息类型 '{msg.get('type')}'，已忽略。")
            continue

        # 获取逻辑客户端ID
        logical_client_idx = msg.get('client_id')
        if logical_client_idx is None:
            logger.warning("收到的消息缺少 client_id，已忽略。")
            continue

        # 检查这个客户端是否是我们正在等待的
        if logical_client_idx in waiting_for_clients:
            logger.info(f"成功收到逻辑客户端 {logical_client_idx} 的训练结果。")
            
            # 从等待集合中移除该客户端
            waiting_for_clients.remove(logical_client_idx)
            logger.info(f"剩余等待客户端: {len(waiting_for_clients)}")

            # --- 以下是正常的处理逻辑 ---
            # 收集模型更新和客户端信息
            model_difference = msg.get('model_difference', {})
            if not model_difference:
                logger.warning(f"客户端 {logical_client_idx} 返回了空的模型更新，已跳过。")
                continue

            client_updates.append(model_difference)
            
            client_info.append({
                'quant_error': msg.get('quant_error', 0.1),
                'energy': msg.get('energy', 0.0),
                'time': msg.get('time', 0.0),
                'samples': msg.get('samples', 0),
                'validation_accuracy': msg.get('validation_accuracy', 0.0),
                'quant_budget': msg.get('quant_budget', 32),
                'battery_level': msg.get('resource', {}).get('battery_level', 0.0)
            })
            
            # 更新全局资源信息
            if 'resource' in msg:
                client_resources[logical_client_idx] = msg['resource']
        else:
            # 收到了一个我们不关心的客户端发来的消息（比如它没被选中，但发来了别的什么）
            logger.warning(f"收到非本轮训练客户端 {logical_client_idx} 的 'train_result' 消息，已忽略。")

    logger.info(f"本轮训练结束，共收到 {len(client_updates)}/{len(selected_clients)} 个有效的客户端训练结果。")
    return client_updates, client_info

if __name__ == '__main__':
    args = args_parser()
    
    # 添加LQMAS特有参数
    args.use_qmix = True
    args.weighted_aggregation = True
    args.quant_budget_options = [4, 8, 16, 32]  # 可选量化精度
    args.exploration_rate = 0.3  # 初始探索率
    args.exploration_decay = 0.95  # 探索率衰减
    args.exploration_min = 0.05  # 最小探索率
    args.qmix_hidden_dim = 64  # QMIX网络隐藏层维度
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 设置日志
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(f"{log_dir}/server_LQMAS_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.info("启动LQMAS服务器")
    logger.info(f"参数配置: {args}")

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    if 'resnet18' in args.model:
        net_glob = ResNet18_entire()
        net_glob.apply(init_weights)
        net_glob.to(args.device)
    if 'vgg' in args.model:
        net_glob = VGG16_entire()
        net_glob.apply(init_weights)
        net_glob.to(args.device)
    if 'resnet8' in args.model:
        net_glob = ResNet8_entire()
        net_glob.apply(init_weights)
        net_glob.to(args.device)
    
    net_glob.apply(init_weights)
    net_glob.to(args.device)
    
    # 初始化QMIX控制器
    # 观察空间: [电量, 计算能力, 网络状况, 数据大小, 验证精度, 训练时间]
    obs_dim = 6  
    # 动作空间: [不参与, 8位量化, 16位量化, 32位量化]
    action_dim = len(args.quant_budget_options) + 1  
    
    qmix_controller = QMIXController(
        num_agents=args.num_users,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.qmix_hidden_dim,
        device=args.device
    )
    
    # 准备存储结果
    summary_acc_test_collect = []
    time_collect = []
    energy_collect = []
    reward_history = []
    
    current_exploration_rate = args.exploration_rate
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        logger.info(f"使用CUDA: {torch.cuda.get_device_name(0)}")

    num_device = int(args.num_users / Reuse_ratio)
    m = max(int(args.frac * num_device), 1)

    # 初始化连接处理器
    connectHandler = ConnectHandler(num_device, args.HOST, args.POST)

    # 创建并分发验证集
    client_val_indices = create_validation_set(dataset_test, args.num_users)
    client_resources = broadcast_validation_set(connectHandler, client_val_indices, args)

    # 主训练循环
    prev_global_acc = 0
    for iter in range(args.epochs):
        logger.info(f"============= 轮次 {iter} =============")
        round_start_time = time.time()
        
        # 阶段1: 发送全局模型给所有客户端进行验证
        validation_results = broadcast_model_for_validation(connectHandler, net_glob, dict_users, iter, args)

        # 阶段2: 基于验证结果使用QMIX选择客户端和分配量化精度
        if iter > 0 and iter % 5 == 0:
            # 每5轮训练一次QMIX网络
            logger.info("训练QMIX网络")
            qmix_controller.train(epochs=10, batch_size=32)
        
        # 获取当前全局状态
        global_state = [prev_global_acc, iter/args.epochs]
        
        # 基于QMIX选择客户端和分配量化精度
        selected_clients, client_quant_budgets, observations, actions = select_clients_with_qmix(
            qmix_controller, 
            client_resources, 
            dict_users, 
            args, 
            global_state, 
            validation_results,
            current_exploration_rate
        )
        
        # 更新探索率
        current_exploration_rate = max(
            args.exploration_min, 
            current_exploration_rate * args.exploration_decay
        )
        
        # 阶段3: 向选定的客户端发送模型和量化配置
        client_updates, client_info = train_selected_clients(
            connectHandler,
            net_glob,
            dict_users,
            selected_clients,
            client_quant_budgets,
            iter,
            args
        )
        
        # 阶段4: 聚合模型
        if len(client_updates) > 0:
            # 使用改进的加权聚合方法
            w_glob = weighted_aggregation(
                client_weights=None,
                client_updates=client_updates,
                client_info=client_info, 
                global_model=net_glob,
                logger=logger
            )
            
            net_glob.load_state_dict(w_glob)
            
            # 评估全局模型
            current_global_acc = summary_evaluate(
                copy.deepcopy(net_glob).to(args.device),
                dataset_test, args.device
            ) * 100
        
        current_time = time.time()
        round_time = current_time - round_start_time
        
        # 计算本轮总能耗
        client_energy = {}
        for idx, info in enumerate(client_info):
            client_energy[idx] = info.get('energy', 0.0)
        round_energy = sum(client_energy.values())
        
        # 计算奖励
        successful_ratio = len(client_info) / len(selected_clients) if selected_clients else 0
        reward = calculate_reward(
            prev_acc=prev_global_acc,
            current_acc=current_global_acc,
            energy_consumption=round_energy,
            total_time=round_time,
            successful_clients_ratio=successful_ratio
        )
        
        # 存储经验到QMIX控制器
        if iter > 0:  # 第一轮没有前一轮的状态，所以不存储
            # 收集下一个状态的观察
            next_observations = []
            for i in range(args.num_users):
                # 默认值
                obs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                
                # 如果有历史数据则使用实际值
                if i % num_device in client_resources:
                    res = client_resources[i % num_device]
                    obs[0] = res.get('battery_level', 0.5) / 100.0
                    obs[1] = res.get('computing_power', 0.5) / 10.0
                    obs[2] = res.get('bandwidth', 0.5) / 10.0
                    obs[3] = len(dict_users[i]) / 1000.0
                    obs[4] = res.get('validation_acc', 0.5)
                    obs[5] = res.get('training_time', 30.0) / 60.0
                
                next_observations.append(obs)
            
            next_global_state = [current_global_acc, (iter+1)/args.epochs]
            
            # 存储转换 (s,a,r,s')
            qmix_controller.store_transition(
                observations, global_state, actions, 
                reward, next_observations, next_global_state, 
                done=(iter == args.epochs-1)
            )
            
            reward_history.append(reward)
            logger.info(f"奖励值: {reward:.4f}")
        
        # 更新上一轮精度记录
        prev_global_acc = current_global_acc
        
        # 记录结果
        summary_acc_test_collect.append(current_global_acc)
        time_collect.append(current_time)
        energy_collect.append(round_energy)
        
        logger.info("====================== LQMAS SERVER ==========================")
        logger.info(f' Test: Round {iter:3d}, Avg Accuracy {current_global_acc:.3f}%, Energy {round_energy:.3f}J, Time {round_time:.3f}s')
        logger.info(f' 累计奖励: {sum(reward_history):.4f}, 平均奖励: {np.mean(reward_history) if reward_history else 0:.4f}')
        logger.info("=============================================================")

    # 保存最终QMIX模型
    qmix_controller.save("./checkpoints/qmix_final.pt")
    
    # 保存结果
    save_result(summary_acc_test_collect, 'test_acc', args)
    save_result(time_collect, 'time', args)
    save_result(energy_collect, 'energy', args)
    save_result(reward_history, 'reward', args)
    
    logger.info("LQMAS服务器训练完成")
