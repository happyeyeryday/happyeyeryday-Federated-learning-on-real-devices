import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .scalefl_modelutils import Scaler, conv3x3


class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes, num_conv_layers=3, reduction=1, scale=1., trs=True):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.reduction = reduction
        self.scale = scale

        if scale < 1:
            scaler = Scaler(scale)
        else:
            scaler = nn.Identity()

        if reduction == 1:
            conv_list = [conv3x3(in_planes, in_planes) for _ in range(num_conv_layers)]
        else:
            conv_list = [conv3x3(in_planes, int(in_planes/reduction))]
            in_planes = int(in_planes/reduction)
            conv_list.extend([conv3x3(in_planes, in_planes) for _ in range(num_conv_layers-1)])

        # 🔥 致命 Bug 修复: track_running_stats=False 导致推理时统计量不稳定
        # 改为 track_running_stats=trs (默认 True)，与 BasicBlock 保持一致
        bn_list = [nn.BatchNorm2d(in_planes, track_running_stats=trs) for _ in range(num_conv_layers)]
        relu_list = [nn.ReLU() for _ in range(num_conv_layers)]
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()

        layers = []
        for i in range(num_conv_layers):
            layers.append(conv_list[i])
            layers.append(scaler)
            layers.append(bn_list[i])
            layers.append(relu_list[i])
        layers.append(avg_pool)
        layers.append(flatten)

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, inp, pred=None):
        output = self.layers(inp)
        output = self.fc(output)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, trs=False, scale=1.):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=trs)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        out = self.bn2(self.scaler(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, trs=False, scale=1.):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        out = F.relu(self.bn2(self.scaler(self.conv2(out))))
        out = self.bn3(self.scaler(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------
# 🔥 修复后的 ResNet 类
# ---------------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, ee_layer_locations=[], scale=1., trs=False, is_full_model=True):
        super(ResNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        if num_classes == 1000: factor = 4
        elif num_classes == 200: factor = 4
        else: factor = 1

        self.scale = scale
        self.in_planes = int(64 * scale * factor)
        self.num_blocks = len(ee_layer_locations) + 1
        self.num_classes = num_classes
        self.trs = trs
        self.is_full_model = is_full_model

        if scale < 1: self.scaler = Scaler(scale)
        else: self.scaler = nn.Identity()

        ee_block_list = []
        ee_layer_list = []

        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)

        # 头部卷积
        if self.num_classes > 100:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, track_running_stats=self.trs)

        # 2. 构建 Layers (通道数统一为 [64, 128, 256, 512])
        
        # Layer 1 (Block 0): 肯定有
        layer1, ee1 = self._make_layer(block, int(64 * scale), layers[0], stride=1,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0])
        
        self.layers = nn.ModuleList([layer1])
        self.ee_classifiers = nn.ModuleList([ee1])

        # Layer 2 (Block 1): 如果有
        if len(layers) > 1:
            layer2, ee2 = self._make_layer(block, int(128 * scale), layers[1], stride=2,
                                           ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1])
            self.layers.append(layer2)
            self.ee_classifiers.append(ee2)

        # [🔥 修复 Nano 崩溃] Layer 3 (Block 2): 必须检查是否存在
        if len(layers) > 2:
            layer3, ee3 = self._make_layer(block, int(256 * scale), layers[2], stride=2,
                                           ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2])
            self.layers.append(layer3)
            self.ee_classifiers.append(ee3)

        # Layer 4 (Block 3): 必须检查是否存在
        if len(layers) > 3:
            layer4, ee4 = self._make_layer(block, int(512 * scale), layers[3], stride=2,
                                           ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 3])
            self.layers.append(layer4)
            self.ee_classifiers.append(ee4)

        # 3. 构建 Linear
        if self.is_full_model:
            # 动态计算 Linear 输入维度 (基础通道64，每层*2)
            final_planes = int(64 * (2**(len(self.layers)-1)) * scale) * block.expansion
            self.linear = nn.Linear(final_planes, num_classes)
        else:
            pass

    def _make_layer(self, block_type, planes, num_block, stride, ee_layer_locations):
        strides = [stride] + [1] * (num_block - 1)
        ee_layer_locations_ = ee_layer_locations + [num_block]
        layers = [[] for _ in range(len(ee_layer_locations_))]
        ee_classifiers = []

        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):
                for _ in range(start_layer, ee_layer_idx):
                    layers[i].append(block_type(self.in_planes, planes, strides[counter], trs=self.trs, scale=self.scale))
                    self.in_planes = planes * block_type.expansion
                    counter += 1
                start_layer = ee_layer_idx

                if ee_layer_idx == 0: num_planes = self.in_planes
                else: num_planes = planes * block_type.expansion

                if i < len(ee_layer_locations_) - 1:
                    ee_classifiers.append(Classifier(num_planes, num_classes=self.num_classes,
                                                     reduction=block_type.expansion, scale=self.scale, trs=self.trs))
        else:
            for i in range(num_block):
                layers[0].append(block_type(self.in_planes, planes, strides[i], trs=self.trs, scale=self.scale))
                self.in_planes = planes * block_type.expansion

        return nn.ModuleList([nn.Sequential(*l) for l in layers]), nn.ModuleList(ee_classifiers)

    # [🔥 修复 3] 修复 IndexError
    @staticmethod
    def find_ee_block_and_layer(layers, layer_idx):
        cum_array = np.cumsum(layers)
        
        # [🔥 修正] 使用默认的 side='left'
        # loc=2 -> Block 0 (正确，Block 0 的末尾)
        # loc=4 -> Block 1 (正确，Block 1 的末尾)
        # loc=6 -> Block 2 (正确，Block 2 的末尾)
        block_idx = np.searchsorted(cum_array, layer_idx)
        
        # 边界检查
        if block_idx >= len(layers):
            return len(layers) - 1, layers[-1]
            
        if block_idx == 0:
            layer = layer_idx
        else:
            layer = layer_idx - cum_array[block_idx - 1]
            
        return block_idx, layer

    def forward(self, x, manual_early_exit_index=0):
        final_out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        if self.num_classes > 100:
            final_out = F.max_pool2d(final_out, kernel_size=3, stride=2, padding=1)
        ee_outs = []
        counter = 0

        while counter < len(self.layers):
            if final_out is not None:
                if manual_early_exit_index > sum([len(ee) for ee in self.ee_classifiers[:counter+1]]):
                    manual_early_exit_index_ = 0
                elif manual_early_exit_index:
                    manual_early_exit_index_ = manual_early_exit_index - sum([len(ee) for ee in self.ee_classifiers[:counter]])
                else:
                    manual_early_exit_index_ = manual_early_exit_index

                final_out = self._block_forward(self.layers[counter], self.ee_classifiers[counter], final_out, ee_outs, manual_early_exit_index_)
            counter += 1

        preds = ee_outs

        if self.is_full_model and final_out is not None:
            out = F.adaptive_avg_pool2d(final_out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            preds.append(out)

        if manual_early_exit_index:
            assert len(preds) == manual_early_exit_index

        return preds

    def _block_forward(self, layers, ee_classifiers, x, outs, early_exit=0):
        for i in range(len(layers)-1):
            x = layers[i](x)
            if outs:
                outs.append(ee_classifiers[i](x, outs[-1]))
            else:
                outs.append(ee_classifiers[i](x))
            if early_exit == i + 1:
                break
        if early_exit == 0:
            final_out = layers[-1](x)
        else:
            final_out = None
        return final_out

# ---------------------------------------------------------------------
# Wrapper 函数
# ---------------------------------------------------------------------

def resnet110_1(args, params):
    return ResNet(BasicBlock, [18, 18, 18], args.num_classes, scale=params.get('scale', 1),
                  trs=args.track_running_stats)

def resnet110_4(args, params):
    ee_layer_locations = params['ee_layer_locations']
    return ResNet(BasicBlock, [18, 18, 18], args.num_classes, ee_layer_locations=ee_layer_locations,
                  scale=params.get('scale', 1), trs=args.track_running_stats)

def resnet18_scalefl(args, params):
    """
    ScaleFL ResNet18 (Standard 4 Layers)
    """
    ee_layer_locations = params.get('ee_layer_locations', [])
    scale = params.get('scale', 1.0)
    
    # 强制 4 个 Block [2, 2, 2, 2]
    return ResNet(
        BasicBlock, 
        layers=[2, 2, 2, 2], 
        num_classes=args.num_classes, 
        ee_layer_locations=ee_layer_locations,
        scale=scale, 
        trs=args.track_running_stats,
        is_full_model=True # Server 初始化默认为 Full
    )