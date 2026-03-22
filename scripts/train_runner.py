import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import subprocess
import csv
from datetime import datetime
from model.SHFL_resnet import shfl_resnet18

# ================= 配置区域 =================
# 你可以根据设备手动修改这里的列表
# Orin 建议: [0, 1, 2, 3] (对应 MAXN, 15W, 30W, 50W)
# Xavier 建议: [0, 2, 3]
# Nano 建议: [0, 1]
TARGET_MODES   = [0,1,2]   # 要测试的 nvpmodel 模式 ID
MODEL_INDICES  = [1, 2, 3, 4]             # 要测试的 SHFL 模型深度（出口编号）

BATCH_SIZE = 32
WARMUP_BATCHES = 20
TEST_BATCHES = 150   # 确保每个模式运行时间 > 30秒，方便功率计采样
COOLDOWN_SECONDS = 20        # 切换 mode 后的冷却时间（秒）
MODEL_COOLDOWN_SECONDS = 5   # 同一 mode 下不同 model 之间的冷却时间（秒）
LOG_FILE = "test_results.csv"
# ===========================================

def set_jetson_hardware(mode_id):
    print(f"\n>>> 正在切换到模式 ID: {mode_id}")
    # 1. 切换功率模式，传入 NO 防止需要重启时脚本卡死等待交互
    result = subprocess.run(
        ["sudo", "nvpmodel", "-m", str(mode_id)],
        input="NO\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    output = result.stdout + result.stderr
    if "Reboot required" in output or "REBOOT" in output.upper():
        print(f"[跳过] 模式 {mode_id} 需要重启才能切换，本次跳过。")
        print(f"       若要测试该模式，请手动执行: sudo nvpmodel -m {mode_id} 并重启后再运行脚本。")
        return False
    # 2. 锁定该模式下的最高频率
    subprocess.run(["sudo", "jetson_clocks"], check=True)
    # 3. 风扇强制全速 (255)
    try:
        subprocess.run("echo 255 | sudo tee /sys/devices/virtual/thermal/cooling_device3/cur_state", shell=True)
    except:
        pass
    print(f">>> 模式 {mode_id} 已锁定，风扇已开启。")
    return True

def run_training(model_idx):
    """
    model_idx: 1~4，对应 SHFL ResNet18 的出口深度
      1 = 只用 mainblock[0]（最浅）
      4 = 用全部 4 个 mainblock（最深）
    """
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 使用 SHFL ResNet18，model_idx 控制使用几个 mainblock
    model = shfl_resnet18(num_classes=10, model_idx=model_idx).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 预热（cuDNN 自动调优）
    print(f"  正在预热 Model-{model_idx}（cuDNN 调优中）...")
    model.train()
    model.eval()
    data_iter = iter(trainloader)
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            inputs, labels = next(data_iter)
            inputs, labels = inputs.cuda(), labels.cuda()
            model(inputs)
    model.train()

    # 正式测量：warmup 全部完成后再记录开始时间，避免 cuDNN 初始化峰值干扰
    torch.cuda.synchronize()  # 等待 warmup GPU 全部结束
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    t_start = time.time()
    print(f"  ▶ Model-{model_idx} 测量开始: {start_time_str}")

    for _ in range(TEST_BATCHES):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            inputs, labels = next(data_iter)

        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        # SHFL Three_ResNet forward 返回单个 logit tensor
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()  # 等待最后一个 batch GPU 全部结束
    t_end = time.time()
    # 额外等待 3 秒，确保功率计能采到 GPU 真正结束前的尾部数据
    time.sleep(3)
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"  ■ Model-{model_idx} 测量结束: {end_time_str}  (训练历时 {t_end - t_start:.1f} 秒)")

    avg_batch_ms = ((t_end - t_start) / TEST_BATCHES) * 1000
    return start_time_str, end_time_str, avg_batch_ms

def main():
    # 每次运行都重写 CSV，避免新旧数据混在一起
    with open(LOG_FILE, 'w') as f:
        f.write("device_model,mode,model_idx,start_time,end_time,avg_batch_time_ms\n")

    # 获取设备型号简称 (Nano/Xavier/Orin)
    with open("/proc/device-tree/model", "r") as f:
        device_model = f.read().strip()

    for m_id in TARGET_MODES:
        print(f"\n{'='*60}")
        print(f"[Mode {m_id}] 开始测试")
        if not set_jetson_hardware(m_id):
            continue  # 该模式需要重启，跳过
        time.sleep(5)  # 等待频率稳定

        for model_idx in MODEL_INDICES:
            print(f"\n  --- Mode {m_id} / Model-{model_idx} ---")
            st, et, batch_ms = run_training(model_idx)

            with open(LOG_FILE, 'a') as f:
                f.write(f"{device_model},{m_id},{model_idx},{st},{et},{batch_ms:.2f}\n")

            print(f"  Mode {m_id} / Model-{model_idx} 完成: {batch_ms:.2f} ms/batch")

            # 同一 mode 下不同 model 之间短暂冷却
            if model_idx != MODEL_INDICES[-1]:
                print(f"  切换下一个模型，冷却 {MODEL_COOLDOWN_SECONDS} 秒...")
                time.sleep(MODEL_COOLDOWN_SECONDS)

        print(f"\n[Mode {m_id}] 全部 Model 测试完成，冷却 {COOLDOWN_SECONDS} 秒...")
        time.sleep(COOLDOWN_SECONDS)

if __name__ == "__main__":
    main()