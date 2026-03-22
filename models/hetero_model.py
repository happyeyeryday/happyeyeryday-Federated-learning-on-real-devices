# models/SplitModel.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 配置区域 (Mock Config)
# 为了不依赖外部 config 文件，我们将必要的配置写在这里
# 你可以根据实际情况修改这些参数 (默认是 CIFAR-10 的 ResNet18 配置)
# ==========================================
cfg = {
    'norm': 'bn',          # 使用 Batch Norm (配合 track 实现 sBN)
    'scale': True,         # 启用 Scaler
    'mask': False,         # 暂时关闭 Mask 逻辑，简化流程
    'global_model_rate': 1.0, 
    'data_shape': [3, 32, 32], # CIFAR-10 尺寸
    'classes_size': 10,        # CIFAR-10 类别数
    'resnet': {
        'hidden_size': [64, 128, 256, 512] # 标准 ResNet 通道数
    }
}

# ==========================================
# 2. 基础组件 (Scaler & Init)
# ==========================================
class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        # 只有在训练时才除以 rate，推理时保持原样
        output = input / self.rate if self.training else input
        return output

def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

# ==========================================
# 3. ResNet 组件 (Block, Bottleneck)
# ==========================================
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, rate, track):
        super(Block, self).__init__()
        # sBN 逻辑: track=False 时，只统计当前 Batch，不更新全局 stats
        if cfg['norm'] == 'bn':
            n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
            n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        else:
            # 简化处理，默认走 BN
            n1 = nn.Identity()
            n2 = nn.Identity()

        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Scaler 逻辑
        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # Pre-activation 结构: Scaler -> BN -> ReLU -> Conv
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride, rate, track):
        super(Bottleneck, self).__init__()
        if cfg['norm'] == 'bn':
            n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
            n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
            n3 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        else:
            n1, n2, n3 = nn.Identity(), nn.Identity(), nn.Identity()

        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = n3
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        
        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out = self.conv3(F.relu(self.n3(self.scaler(out))))
        out += shortcut
        return out

# ==========================================
# 4. 主 ResNet 类
# ==========================================
class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, rate, track):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, rate=rate, track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, rate=rate, track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, rate=rate, track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, rate=rate, track=track)
        
        if cfg['norm'] == 'bn':
            n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion, momentum=None, track_running_stats=track)
        else:
            n4 = nn.Identity()
        self.n4 = n4
        
        if cfg['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()
            
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rate, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # [修改] 改为标准 PyTorch 接口，直接接收 tensor x，而不是 dict input
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        # [修改] 直接返回 tensor，方便计算 Loss
        return out

# ==========================================
# 5. 工厂函数
# ==========================================
def resnet18(model_rate=1, track=False):
    # 使用内部定义的 cfg
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    # 动态计算 Hidden Size
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, scaler_rate, track)
    model.apply(init_param)
    return model

# 如果需要其他模型 (ResNet34/50等)，可以类似复制过来