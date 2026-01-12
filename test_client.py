import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import socket
import sys
import os

# ================= 配置区域 =================
# 接收端的 IP 地址 (你的主机 IP)
SERVER_IP = '192.168.31.165'
SERVER_PORT = 9999

# 训练配置
# 建议与正式实验保持一致，Nano 建议 BS=16 或 32
TEST_BATCH_SIZE = 32
NUM_WORKERS = 0        # Nano 必须为 0，防止 Swap 卡死

# 测试用的模型深度 (SHFL Model-4 代表全量深度，用于测最大负载)
TEST_MODEL_IDX = 4     
# ===========================================


try:
    from utils.get_dataset import *
    from utils.options import args_parser
    from utils.set_seed import set_random_seed
    from utils.FL_utils import *
    # 引入新的 1/4 通道 SHFL 模型
    from models.SHFL_resnet import shfl_resnet18 
except ImportError as e:
    print("❌ 导入错误: 请确保脚本在 server_fedavg 根目录下运行")
    print(f"详细错误: {e}")
    sys.exit(1)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test_training_one_epoch(device, args, cid, dataset_train, dict_users):
    """
    测试该 Client 在其特定数据量下跑完 1 个 Epoch 的时间
    
    Args:
        dataset_train: 训练数据集（复用，不重复加载）
        dict_users: 数据划分字典（复用，保证一致性）
    """
    print(f"\n[测试] 训练性能测试 (Client ID: {cid})")
    print("-" * 60)
    
    # 1. 获取该 CID 的数据索引
    if cid not in dict_users:
        print(f"❌ 错误: Client ID {cid} 不在数据分配列表中！")
        return None, None, None

    idxs = dict_users[cid]
    data_len = len(idxs)
    print(f"📊 本机分配数据量: {data_len} 张图片")
    
    # 构建 DataLoader
    local_dataset = DatasetSplit(dataset_train, idxs)
    dtLoader = DataLoader(local_dataset, batch_size=TEST_BATCH_SIZE, 
                          shuffle=True, num_workers=NUM_WORKERS)
    
    # 2. 准备模型 (SHFL ResNet18, 1/4通道版)
    model = shfl_resnet18(num_classes=args.num_classes, model_idx=TEST_MODEL_IDX).to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_func = nn.CrossEntropyLoss()
    
    # 3. 开始训练
    print(f"🚀 开始训练 1 个 Epoch (BatchSize={TEST_BATCH_SIZE})...")
    
    # 强制同步 GPU 时间
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(dtLoader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        # SHFL 兼容性处理 (取最后一个出口)
        output = outputs[-1] if isinstance(outputs, list) else outputs
        
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        
        # 可选：打印进度
        # if batch_idx % 10 == 0:
        #     print(f"\rBatch {batch_idx}/{len(dtLoader)}", end="")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    duration = time.time() - start_time
    
    print(f"\n✅ 训练完成！")
    print(f"⏱️  耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"⚡ 速度: {data_len/duration:.2f} img/s")
    
    return model, data_len, duration # 返回模型、数据量、耗时

def test_communication(model):
    """
    测试发送模型到指定 IP 的耗时
    """
    print(f"\n[测试 2] 通讯性能测试 (Target: {SERVER_IP}:{SERVER_PORT})")
    print("-" * 60)
    
    # 1. 序列化模型
    print("📦 正在序列化模型...")
    model_cpu = model.cpu()
    state_dict = model_cpu.state_dict()
    
    # pickle 序列化
    serialized_data = pickle.dumps(state_dict)
    data_size = len(serialized_data)
    data_size_mb = data_size / (1024 * 1024)
    print(f"📊 模型大小: {data_size_mb:.2f} MB")
    
    # 2. 发送数据
    print(f"🚀 开始发送数据...")
    start_time = time.time()
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10) # 10秒连接超时
            s.connect((SERVER_IP, SERVER_PORT))
            s.sendall(serialized_data)
            
        duration = time.time() - start_time
        print(f"✅ 发送完成！")
        print(f"⏱️  耗时: {duration:.2f} 秒")
        print(f"⚡ 平均带宽: {data_size_mb * 8 / duration:.2f} Mbps")
        
    except ConnectionRefusedError:
        print("❌ 连接被拒绝！请检查：")
        print(f"   1. 主机 ({SERVER_IP}) 是否运行了接收脚本？")
        print(f"   2. 防火墙是否允许端口 {SERVER_PORT}？")
    except Exception as e:
        print(f"❌ 发送失败: {e}")

def warmup(device, args):
    """
    热身函数：让 GPU 和 CuDNN 进入状态
    """
    print("\n🔥 正在进行 GPU 热身 (Warmup)...")
    # 创建一个临时模型和数据
    model = shfl_resnet18(num_classes=args.num_classes, model_idx=TEST_MODEL_IDX).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    
    # 虚拟数据 (Batch Size 必须和测试时一致)
    dummy_input = torch.randn(TEST_BATCH_SIZE, 3, 32, 32).to(device)
    dummy_label = torch.randint(0, 10, (TEST_BATCH_SIZE,)).to(device)
    
    # 跑 10 个 Batch
    start = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        output = outputs[-1] if isinstance(outputs, list) else outputs
        loss = loss_func(output, dummy_label)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    
    print(f"✅ 热身完成 (耗时 {time.time()-start:.2f}s)。现在的测试数据才是真实的。")

if __name__ == '__main__':
    # 初始化参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # [🔥 关键] 设置随机种子（必须在 get_dataset 之前）
    set_random_seed(args.seed)
    
    print("=" * 70)
    print("🔬 Nano 设备性能测试 (Client ID: 0-6)")
    print("=" * 70)
    print(f"当前设备: {args.device}")
    print(f"批大小: {TEST_BATCH_SIZE}")
    print(f"模型深度: Model-{TEST_MODEL_IDX}")
    print(f"随机种子: {args.seed}")
    print()
    
    # [🔥 关键] 只加载一次数据划分，保证所有测试使用相同的划分
    print("📦 正在加载数据分布（只加载一次）...")
    dataset_train, _, dict_users = get_dataset(args)
    print(f"✅ 数据加载完成 (共 {args.num_users} 个客户端)")
    
    # 收集所有测试结果
    results = []
    trained_model = None

    warmup(args.device, args)
    
    # 循环测试 Client 0-6
    for cid in range(9,10):
        model, data_len, duration = test_training_one_epoch(
            args.device, args, cid, dataset_train, dict_users
        )
        
        if model is not None and data_len is not None and duration is not None:
            results.append({
                'cid': cid,
                'data_len': data_len,
                'duration': duration,
                'speed': data_len / duration
            })
            # 保存最后一个模型用于通讯测试
            if trained_model is None:
                trained_model = model
        
        # 清理显存
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 打印汇总表格
    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"{'Client ID':<12} {'数据量':<12} {'耗时(秒)':<15} {'速度(img/s)':<15}")
    print("-" * 70)
    
    total_time = 0
    for r in results:
        print(f"{r['cid']:<12} {r['data_len']:<12} {r['duration']:<15.2f} {r['speed']:<15.2f}")
        total_time += r['duration']
    
    print("-" * 70)
    print(f"{'平均':<12} {'':<12} {total_time/len(results):<15.2f} {sum(r['speed'] for r in results)/len(results):<15.2f}")
    print(f"{'总计':<12} {sum(r['data_len'] for r in results):<12} {total_time:<15.2f}")
    print("=" * 70)
    
    # 执行通讯测试（只测一次）
    if trained_model:
        test_communication(trained_model)