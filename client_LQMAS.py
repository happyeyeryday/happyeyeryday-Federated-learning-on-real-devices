import torch
from torch import nn
import torch.nn.functional as F
import time
import random
import numpy as np
import copy
import os
import json
from loguru import logger
from collections import OrderedDict

# 导入jtop用于获取Jetson设备功率信息
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    logger.warning("jtop库不可用，将使用模拟的功率数据")

from utils.ConnectHandler_client import ConnectHandler
from utils.get_dataset import *
from utils.FL_utils import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.SplitModel import ResNet18_client_side, ResNet18_server_side, VGG16_client_side, VGG16_server_side, \
    ResNet8_client_side, ResNet8_server_side, ResNet18_entire, VGG16_entire, ResNet8_entire

# 全局变量用于存储验证集
validation_indices = None
validation_loader = None
Reuse_ratio = 2

def setup_validation_set(indices, dataset_test, args, client_id):
    """设置验证数据集"""
    global validation_loaders_map

    # 创建验证集数据加载器
    validation_dataset = DatasetSplit(dataset_test, indices)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0
    )

    # 存储到对应的逻辑客户端映射中
    validation_loaders_map[client_id] = validation_loader

    logger.info(f"为逻辑客户端 {client_id} 设置验证集，包含 {len(indices)} 个样本")
    return len(indices)

def validate_model(net, args, client_id):
    """在验证集上评估模型"""
    global validation_loaders_map

    if client_id not in validation_loaders_map:
        logger.error(f"逻辑客户端 {client_id} 的验证集未设置")
        return 0.0

    validation_loader = validation_loaders_map[client_id]

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            outputs = net(images)['output']
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    logger.info(f"逻辑客户端 {client_id} 验证集准确率: {accuracy:.2f}%")
    return accuracy

class ClientResources:
    """客户端资源状态类"""
    def __init__(self, initial_energy=20000.0):  # 默认20000焦耳，约等于一个小型移动设备电池
        # 初始化资源状态
        self.battery_level = initial_energy  # 焦耳单位，不再使用百分比
        self.computing_power = self._get_computing_power()  # 计算能力 (FLOPS)
        self.bandwidth = self._get_network_bandwidth()  # 网络带宽 (Mbps)
        self.memory_available = self._get_memory_available()  # 可用内存 (MB)
        self.training_power = self._get_training_power()  # 训练功率 (W)
        self.transmission_power = self._get_transmission_power()  # 传输功率 (W)
        self.validation_acc = 0.0  # 验证准确率
        self.training_time = 0.0  # 上一轮训练时间

        # 创建jtop对象用于获取功率信息
        if JTOP_AVAILABLE:
            self.jetson = jtop()
            try:
                self.jetson.start()
                logger.info("成功初始化jtop监控")
            except Exception as e:
                logger.error(f"初始化jtop失败: {e}")
                self.jetson = None

    def update(self, energy_consumption=0.0, training_time=0.0):
        """更新资源状态"""
        # 直接减少电池电量(焦耳)，不再需要除以10.0的转换
        self.battery_level = max(0, self.battery_level - energy_consumption)

        # 更新训练时间记录
        if training_time > 0:
            self.training_time = training_time

        # 更新资源信息
        self.computing_power = self._get_computing_power()
        self.bandwidth = self._get_network_bandwidth()
        self.memory_available = self._get_memory_available()
        self.training_power = self._get_training_power()
        self.transmission_power = self._get_transmission_power()

    def _get_computing_power(self):
        """估计计算能力"""
        if JTOP_AVAILABLE and hasattr(self, 'jetson') and self.jetson and self.jetson.ok():
            try:
                # 尝试多种方式获取CPU信息
                cpu_usage = None

                # 方式1: 直接获取CPU使用率
                try:
                    cpu_info = self.jetson.cpu
                    if hasattr(cpu_info, 'total'):
                        cpu_usage = cpu_info.total
                    elif isinstance(cpu_info, dict):
                        # 尝试不同的键名
                        for key in ['usage', 'total', 'user']:
                            if key in cpu_info:
                                if isinstance(cpu_info[key], (int, float)):
                                    cpu_usage = cpu_info[key]
                                    break
                                elif isinstance(cpu_info[key], dict):
                                    # 如果是字典，尝试获取user + system
                                    user = cpu_info[key].get('user', 0)
                                    system = cpu_info[key].get('system', 0)
                                    cpu_usage = user + system
                                    break
                except Exception:
                    pass

                # 方式2: 使用频率作为计算能力指标
                if cpu_usage is None:
                    try:
                        logger.info(f"方式2获取CPU信息 {e}")
                        # 使用CPU频率来估算计算能力
                        freq_info = self.jetson.cpu
                        if hasattr(freq_info, 'frequency'):
                            # 频率越高，计算能力越强
                            max_freq = 1900  # Jetson Nano 最大频率约1.9GHz
                            current_freq = freq_info.frequency
                            cpu_usage = (current_freq / max_freq) * 100
                    except Exception:
                        pass

                # 转换为0-10范围
                if cpu_usage is not None:
                    # 确保在合理范围内，并转换为计算能力指标
                    cpu_usage = max(0.0, min(100.0, cpu_usage))
                    computing_power = (cpu_usage / 100.0) * 7.0 + 3.0  # 映射到3-10范围
                    return computing_power

            except Exception as e:
                logger.error(f"获取CPU信息失败: {e}")

        # 回退到模拟值
        return random.uniform(3.0, 8.0)

    def _get_network_bandwidth(self):
        """估计网络带宽"""
        # 模拟不同的网络条件 (Mbps)
        return random.uniform(1.0, 10.0)

    def _get_memory_available(self):
        """获取可用内存 (MB)"""
        if JTOP_AVAILABLE and hasattr(self, 'jetson') and self.jetson and self.jetson.ok():
            try:
                # jtop 4.x 版本的访问方式
                memory_info = self.jetson.memory
                if isinstance(memory_info, dict):
                    # 方式1: 直接字典访问
                    return memory_info.get('tot', 4096) - memory_info.get('used', 0)
                else:
                    # 方式2: 对象属性访问
                    return memory_info.tot - memory_info.used
            except Exception as e:
                logger.info(f"jtop获取内存信息失败: {e},尝试stats方案")
                # 尝试备用访问方式
                try:
                    # 备用方式：通过stats获取
                    stats = self.jetson.stats
                    memory = stats.get('memory', {})
                    return memory.get('tot', 4096) - memory.get('used', 0)
                except Exception as e2:
                    logger.error(f"备用内存获取方式也失败: {e2},回退模拟值")

        # 回退到模拟值
        return 2048.0  # 默认2GB

    def _get_training_power(self):
        """获取训练功率 (W)"""
        if JTOP_AVAILABLE and hasattr(self, 'jetson') and self.jetson and self.jetson.ok():
            try:
                # 正确访问总功率，单位为mW，转换为W
                return self.jetson.power['tot']['power'] / 1000.0
            except Exception as e:
                logger.error(f"获取功率信息失败: {e}")

        # 回退到模拟值
        return random.uniform(3.0, 8.0)

    def _get_transmission_power(self):
        """获取传输功率 (W)"""
        if JTOP_AVAILABLE and hasattr(self, 'jetson') and self.jetson and self.jetson.ok():
            try:
                # 使用总功率的30%作为传输功率估计
                return (self.jetson.power['tot']['power'] / 1000.0) * 0.3
            except Exception as e:
                logger.error(f"获取功率信息失败: {e}")

        # 回退到模拟值
        return random.uniform(1.0, 3.0)

    def to_dict(self):
        """转换为字典表示"""
        return {
            'battery_level': self.battery_level,
            'computing_power': self.computing_power,
            'bandwidth': self.bandwidth,
            'memory_available': self.memory_available,
            'training_power': self.training_power,
            'transmission_power': self.transmission_power,
            'validation_acc': self.validation_acc,
            'training_time': self.training_time
        }

    def __del__(self):
        """析构函数，确保jtop关闭"""
        if JTOP_AVAILABLE and hasattr(self, 'jetson') and self.jetson:
            try:
                self.jetson.close()
            except:
                pass

def uniform_quantize(tensor, bits=8):
    """对张量进行均匀量化（在GPU上高效执行）"""
    if bits == 32:
        return tensor  # 不量化

    # 计算量化范围
    qmin = -2**(bits-1)
    qmax = 2**(bits-1) - 1

    # 在GPU上直接计算，因为现在处理的是单层张量，不会超时
    scale = torch.max(torch.abs(tensor)) / qmax
    scale = torch.max(scale, torch.tensor(1e-8, device=tensor.device))  # 防止除零

    # 量化
    tensor_q = torch.round(tensor / scale)
    tensor_q = torch.clamp(tensor_q, qmin, qmax)

    # 反量化
    tensor_dq = tensor_q * scale

    return tensor_dq

def train(net, dataset, args, idxs_list, device, quant_budget=32):
    """本地训练函数，返回模型更新而非完整模型"""
    net.train()

    # 记录开始时间
    start_time = time.time()

    # 初始化优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # 保存初始模型权重
    initial_weights = copy.deepcopy(net.state_dict())

    # --- 安全且高效的数据加载方式 ---
    # 1. 创建一个只包含本客户端数据的子数据集
    train_subset = torch.utils.data.Subset(dataset, idxs_list)

    # 2. 将子数据集传递给DataLoader
    # DataLoader会自动处理从硬盘到内存再到GPU的批次数据流
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.local_bs, shuffle=True, num_workers=0
    )

    # # 训练数据
    # if args.dataset == 'mnist':
    #     images = dataset.data[idxs_list].float().unsqueeze(1).to(device)
    #     labels = dataset.targets[idxs_list].long().to(device)
    #     train_data = [(image, label) for image, label in zip(images, labels)]
    # else:
    #     images = torch.stack([dataset[i][0] for i in idxs_list]).to(device)
    #     labels = torch.tensor([dataset[i][1] for i in idxs_list]).long().to(device)
    #     train_data = [(image, label) for image, label in zip(images, labels)]

    # # 使用一个对Jetson Nano更安全的批处理大小
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.local_bs, shuffle=True, num_workers=0
    # )

    # 训练
    for epoch in range(args.local_ep):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = net(images)['output']
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()

    # 计算模型更新和量化误差
    quant_error = 0.0
    total_params = 0
    model_difference = OrderedDict()

    for name, param in net.state_dict().items():
        # 计算参数更新
        update = param - initial_weights[name]

        # 量化更新
        quantized_update = uniform_quantize(update, bits=quant_budget)

        # 保存量化后的更新
        model_difference[name] = quantized_update

        # 计算量化误差
        error = torch.sum((quantized_update - update)**2).item()
        total = torch.sum(update**2).item()

        if total > 0:
            quant_error += error / total
            total_params += 1

    # 平均量化误差
    if total_params > 0:
        quant_error /= total_params

    # 计算训练时间
    training_time = time.time() - start_time

    # 计算能耗 (基于训练时间和功率)
    energy_consumption = training_time * client_resources.training_power

    return model_difference, quant_error, training_time, energy_consumption, len(train_subset)

def validate(net, dataset, args, device):
    """在验证集上评估模型"""
    net.eval()

    # 随机选择一小部分数据作为验证集
    val_size = min(1000, len(dataset))
    val_indices = np.random.choice(len(dataset), val_size, replace=False)

    val_images = torch.stack([dataset[i][0] for i in val_indices]).to(device)
    val_labels = torch.tensor([dataset[i][1] for i in val_indices]).long().to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        outputs = net(val_images)['output']
        _, predicted = torch.max(outputs.data, 1)
        total = val_labels.size(0)
        correct = (predicted == val_labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 设置日志
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(f"{log_dir}/client_LQMAS_{args.CID}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"启动LQMAS客户端 {args.CID}")

    client_resources_map = {}  # 存储每个逻辑客户端的资源状态
    validation_loaders_map = {}  # 存储每个逻辑客户端的验证加载器
    current_client_id = None  # 当前处理的逻辑客户端ID

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # 初始化资源监控
    default_resources = ClientResources()
    logger.info(f"初始资源状态: {json.dumps(default_resources.to_dict(), indent=2)}")

    # 初始化连接
    ID = args.CID
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    logger.info("等待服务器连接...")

    # 在主循环部分修改
    pending_train_requests = []  # 存储待处理的训练请求

    while True:
        logger.info(f"回到主循环顶部，准备接收来自服务器的新指令 (当前逻辑客户端: {current_client_id})...")
        msg = connectHandler.receiveFromServer()

        if msg['type'] == 'train':
            # 收集训练请求而不是立即处理
            pending_train_requests.append(msg)
            client_id = msg.get('client_id', ID)
            logger.info(f"收到逻辑客户端 {client_id} 的训练请求，加入待处理队列")

            # 检查是否收到了所有预期的训练请求
            # 这里需要一个机制来确定何时开始处理（比如超时或收到特定信号）
            if len(pending_train_requests) >= Reuse_ratio:
                logger.info(f"达到批处理阈值，开始处理 {len(pending_train_requests)} 个训练请求")
                msg = {'type': 'process_pending_trains'}
            else:
                continue  # 继续等待更多请求

        elif msg['type'] == 'process_pending_trains' and pending_train_requests:
            # 批量处理所有训练请求
            logger.info(f"开始批量处理 {len(pending_train_requests)} 个训练请求")

            # 批量处理所有训练请求
            for train_msg in pending_train_requests:
                logger.info(f"处理训练请求: 逻辑客户端 {train_msg.get('client_id')}")

                # 获取消息中的客户端ID
                current_client_id = train_msg.get('client_id', ID)

                # 确保当前客户端有资源对象
                if current_client_id not in client_resources_map:
                    client_resources_map[current_client_id] = copy.copy(default_resources)
                    logger.info(f"为逻辑客户端 {current_client_id} 创建资源状态")

                # 获取当前逻辑客户端的资源对象
                client_resources = client_resources_map[current_client_id]

                # 阶段2: 训练阶段 - 接收模型进行量化训练，
                round_idx = train_msg['round']
                logger.info(f"收到轮次 {round_idx} 逻辑客户端 {current_client_id} 的训练请求")

                # 获取量化精度和数据索引
                quant_budget = train_msg.get('quant_budget', 32)
                idxs_list = train_msg['idxs_list']

                logger.info(f"本轮使用 {quant_budget} 位量化")

                # 如果电量不足，则跳过训练
                if client_resources.battery_level < 50:
                    logger.warning("电量不足，跳过训练")
                    response = dict()
                    response['type'] = 'train_result'
                    response['model_difference'] = {}
                    response['quant_error'] = 1.0
                    response['resource'] = client_resources.to_dict()
                    response['client_id'] = current_client_id

                    connectHandler.uploadToServer(response)
                    continue

                # 初始化模型
                net = train_msg['net']
                net.to(args.device)

                # 本地训练并量化模型更新
                model_difference, quant_error, training_time, energy_consumption, num_samples = train(
                    net, dataset_train, args, idxs_list, args.device, quant_budget
                )

                # 在验证集上评估
                val_acc = validate_model(net, args, current_client_id)

                # 更新资源状态
                client_resources.validation_acc = val_acc
                client_resources.update(energy_consumption, training_time)

                # 创建响应
                response = dict()
                response['type'] = 'train_result'
                response['model_difference'] = model_difference
                response['quant_error'] = quant_error
                response['energy'] = energy_consumption
                response['time'] = training_time
                response['samples'] = num_samples
                response['validation_accuracy'] = val_acc
                response['quant_budget'] = quant_budget
                response['resource'] = client_resources.to_dict()
                response['client_id'] = current_client_id

                logger.info(f"准备调用 uploadToServer 将训练结果发送给服务器...")
                # 发送响应
                connectHandler.uploadToServer(response)
                logger.info(f"uploadToServer 调用完成，本轮训练任务处理完毕！")

                logger.info(f"已完成轮次 {round_idx} 逻辑客户端 {current_client_id} 的训练")
                logger.info(f"训练时间: {training_time:.2f}s, 能耗: {energy_consumption:.2f}J")
                logger.info(f"样本数: {num_samples}, 量化误差: {quant_error:.6f}")
                logger.info(f"验证精度: {val_acc:.2f}%, 剩余电量: {client_resources.battery_level:.1f}J")

            # 清空待处理队列
            pending_train_requests.clear()
            logger.info("批量训练处理完成，清空待处理队列")
            continue

        # 对于非训练请求，按原来的逻辑处理
        # 获取消息中的客户端ID
        current_client_id = msg.get('client_id', ID)

        # 确保当前客户端有资源对象
        if current_client_id not in client_resources_map:
            # 为新的逻辑客户端创建资源对象，使用浅拷贝来维护不同的状态
            client_resources_map[current_client_id] = copy.copy(default_resources)
            logger.info(f"为逻辑客户端 {current_client_id} 创建资源状态")

        # 获取当前逻辑客户端的资源对象
        client_resources = client_resources_map[current_client_id]

        if msg['type'] == 'validation_set':
            # 接收验证集索引
            logger.info(f"收到逻辑客户端 {current_client_id} 的验证集索引")
            validation_size = setup_validation_set(msg['validation_indices'], dataset_test, args, current_client_id)

            # 获取初始静息功耗
            idle_power = client_resources._get_training_power()
            logger.info(f"逻辑客户端 {current_client_id} 静息功耗: {idle_power:.2f}W")

            # 返回确认和资源信息
            response = dict()
            response['type'] = 'validation_confirmed'
            response['resource'] = client_resources.to_dict()
            response['client_id'] = current_client_id  # 返回逻辑客户端ID

            connectHandler.uploadToServer(response)
            logger.info(f"已确认逻辑客户端 {current_client_id} 接收验证集，包含 {validation_size} 个样本")

        elif msg['type'] == 'validation':
            # 阶段1: 验证阶段 - 接收模型进行验证
            round_idx = msg['round']
            logger.info(f"收到轮次 {round_idx} 逻辑客户端 {current_client_id} 的模型验证请求")

            # 获取数据索引
            idxs_list = msg['idxs_list']

            # 如果电量不足，则跳过验证
            if client_resources.battery_level < 50:
                logger.warning(f"逻辑客户端 {current_client_id} 电量不足 ({client_resources.battery_level:.1f}J)，跳过验证")
                response = dict()
                response['type'] = 'validation_result'
                response['accuracy'] = 0.0
                response['resource'] = client_resources.to_dict()
                response['client_id'] = current_client_id

                connectHandler.uploadToServer(response)
                continue

            # 初始化模型
            net = msg['net']
            net.to(args.device)

            # 在训练集上训练1轮
            start_time = time.time()
            train_loader = DataLoader(
                DatasetSplit(dataset_train, idxs_list),
                batch_size=args.local_bs,
                shuffle=True,
                num_workers=0
            )

            # 记录训练功耗
            if JTOP_AVAILABLE and hasattr(client_resources, 'jetson') and client_resources.jetson:
                try:
                    start_power = client_resources.jetson.power['tot']['power'] / 1000.0  # mW to W
                except Exception as e:
                    logger.error(f"获取起始功率失败: {e}")
                    start_power = client_resources._get_training_power()
            else:
                start_power = client_resources._get_training_power()

            # 训练过程
            net.train()
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                fx = net(images)['output']
                loss = F.cross_entropy(fx, labels)
                loss.backward()
                optimizer.step()

            training_time = time.time() - start_time

            # 记录训练后功耗
            if JTOP_AVAILABLE and hasattr(client_resources, 'jetson') and client_resources.jetson:
                try:
                    end_power = client_resources.jetson.power['tot']['power'] / 1000.0
                except Exception as e:
                    logger.error(f"获取结束功率失败: {e}")
                    end_power = client_resources._get_training_power()
            else:
                end_power = client_resources._get_training_power()

            # 计算平均功耗和总能耗
            avg_power = (start_power + end_power) / 2
            energy_consumption = avg_power * training_time

            # 在验证集上评估
            val_acc = validate_model(net, args, current_client_id)

            # 更新资源状态
            client_resources.validation_acc = val_acc
            client_resources.training_time = training_time
            client_resources.update(energy_consumption, training_time)

            # 创建响应
            response = dict()
            response['type'] = 'validation_result'
            response['accuracy'] = val_acc
            response['training_time'] = training_time
            response['energy'] = energy_consumption
            response['resource'] = client_resources.to_dict()
            response['client_id'] = current_client_id

            # 发送响应
            connectHandler.uploadToServer(response)

            logger.info(f"已完成轮次 {round_idx} 逻辑客户端 {current_client_id} 的验证")
            logger.info(f"验证精度: {val_acc:.2f}%, 训练时间: {training_time:.2f}s, 能耗: {energy_consumption:.2f}J")
        else:
            logger.warning(f"收到未知类型的消息: {msg['type']}")


