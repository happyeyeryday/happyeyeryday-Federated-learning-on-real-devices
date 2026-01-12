import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np

__all__ = ['_200_3_ResNet18_byot_ALL', \
           '_200_3_ResNet18_byot', \
           '_10_3_ResNet18_byot_ALL', \
           '_10_3_ResNet18_byot', \
           '_100_3_ResNet18_byot_ALL', \
           '_100_3_ResNet18_byot', \
           '_10_1_ResNet18_byot', \
           '_10_1_ResNet18_byot_ALL', \
           'ResNet50_byot']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Three_ResNet(nn.Module):
    def __init__(self, block, num_classes, branch_layers=[], planes=[], strides=[], num_blocks=[], is_bias=True,
                 n_models=4):
        super(Three_ResNet, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        mainblocks = []

        self.n_models = n_models
        self.in_plane = planes[0]

        for i in range(n_models):
            mainblocks.append(self._make_resblocklayer(block, planes[i], num_blocks[i], strides[i]))

        self.mainblocks = nn.ModuleList(mainblocks)
        ###########
        self.bottleneck = self._make_bottlenecklayer(block, num_classes, branch_layers[n_models - 1], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)
        #############
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_resblocklayer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        return nn.Sequential(*layers)

    def _make_bottlenecklayer(self, block, num_classes, branch_layers, stride):
        layers = []
        input_planes = self.in_planes
        for i in range(branch_layers):
            layers.append(block(input_planes, input_planes * 2, stride))
            input_planes = input_planes * 2

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # 全局
        if y == None:

            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            # 各客户端
            for idx in range(self.n_models):
                out = self.mainblocks[idx](out)

            features = self.bottleneck(out)
            x = self.avgpool(features)
            x = x.view(x.size(0), -1)

            output = self.fc(x)
        else:
            print("error")

        return output


class One_ResNet(nn.Module):
    def __init__(self, block, num_classes, branch_layers=[], planes=[], strides=[], num_blocks=[], is_bias=True,
                 n_models=4):
        super(One_ResNet, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(1, 64)
        self.bn1 = nn.BatchNorm2d(64)
        mainblocks = []

        self.n_models = n_models
        self.in_plane = planes[0]

        for i in range(n_models):
            mainblocks.append(self._make_resblocklayer(block, planes[i], num_blocks[i], strides[i]))

        self.mainblocks = nn.ModuleList(mainblocks)
        ###########
        self.bottleneck = self._make_bottlenecklayer(block, num_classes, branch_layers[n_models - 1], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)
        #############
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_resblocklayer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        return nn.Sequential(*layers)

    def _make_bottlenecklayer(self, block, num_classes, branch_layers, stride):
        layers = []
        input_planes = self.in_planes
        for i in range(branch_layers):
            layers.append(block(input_planes, input_planes * 2, stride))
            input_planes = input_planes * 2

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # 全局
        if y == None:

            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            # 各客户端
            for idx in range(self.n_models):
                out = self.mainblocks[idx](out)

            features = self.bottleneck(out)
            x = self.avgpool(features)
            x = x.view(x.size(0), -1)

            output = self.fc(x)
        else:
            print("error")

        return output


class Three_ResNet_ALL(nn.Module):
    def __init__(self, block, num_classes, branch_layers=[], planes=[], strides=[], num_blocks=[], is_bias=True,
                 n_models=4):
        super(Three_ResNet_ALL, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        mainblocks = []
        bottlenecks = []
        fcs = []
        self.n_models = n_models
        self.in_plane = planes[0]

        for i in range(n_models):
            mainblocks.append(self._make_resblocklayer(block, planes[i], num_blocks[i], strides[i]))
            bottlenecks.append(self._make_bottlenecklayer(block, num_classes, branch_layers[i], 2))
            fcs.append(nn.Linear(512 * block.expansion, num_classes, bias=is_bias))
        self.mainblocks = nn.ModuleList(mainblocks)
        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.fcs = nn.ModuleList(fcs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_resblocklayer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        return nn.Sequential(*layers)

    def _make_bottlenecklayer(self, block, num_classes, branch_layers, stride):
        layers = []
        input_planes = self.in_planes
        for i in range(branch_layers):
            layers.append(block(input_planes, input_planes * 2, stride))
            input_planes = input_planes * 2

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # 全局
        if y == None:
            embedding = 0.
            features_head = [0.] * (self.n_models)
            features_end = [0.] * (self.n_models)
            output = [0.] * self.n_models

            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            # 各客户端
            for idx in range(self.n_models):
                out = self.mainblocks[idx](out)
                features_head[idx] = out
                features_end[idx] = self.bottlenecks[idx](out)
                x = self.avgpool(features_end[idx])
                x = x.view(x.size(0), -1)
                embedding = x
                output[idx] = self.fcs[idx](x)
        else:
            print("error")

        return output, features_end, embedding


class One_ResNet_ALL(nn.Module):
    def __init__(self, block, num_classes, branch_layers=[], planes=[], strides=[], num_blocks=[], is_bias=True,
                 n_models=4):
        super(One_ResNet_ALL, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(1, 64)
        self.bn1 = nn.BatchNorm2d(64)
        mainblocks = []
        bottlenecks = []
        fcs = []
        self.n_models = n_models
        self.in_plane = planes[0]

        for i in range(n_models):
            mainblocks.append(self._make_resblocklayer(block, planes[i], num_blocks[i], strides[i]))
            bottlenecks.append(self._make_bottlenecklayer(block, num_classes, branch_layers[i], 2))
            fcs.append(nn.Linear(512 * block.expansion, num_classes, bias=is_bias))
        self.mainblocks = nn.ModuleList(mainblocks)
        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.fcs = nn.ModuleList(fcs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_resblocklayer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        return nn.Sequential(*layers)

    def _make_bottlenecklayer(self, block, num_classes, branch_layers, stride):
        layers = []
        input_planes = self.in_planes
        for i in range(branch_layers):
            layers.append(block(input_planes, input_planes * 2, stride))
            input_planes = input_planes * 2

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # 全局
        if y == None:
            embedding = 0.
            features_head = [0.] * (self.n_models)
            features_end = [0.] * (self.n_models)
            output = [0.] * self.n_models

            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
            # 各客户端
            for idx in range(self.n_models):
                out = self.mainblocks[idx](out)
                features_head[idx] = out
                features_end[idx] = self.bottlenecks[idx](out)
                x = self.avgpool(features_end[idx])
                x = x.view(x.size(0), -1)
                embedding = x
                output[idx] = self.fcs[idx](x)
        else:
            print("error")

        return output, features_end, embedding


def _200_3_ResNet18_byot_ALL(pretrained=False, **kwargs):
    return Three_ResNet_ALL(PreActBlock, num_classes=200, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                            branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], **kwargs, n_models=4)


def _200_3_ResNet18_byot(n_models):
    return Three_ResNet(PreActBlock, num_classes=200, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                        branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], n_models=n_models)


def _10_3_ResNet18_byot_ALL(pretrained=False, **kwargs):
    return Three_ResNet_ALL(PreActBlock, num_classes=10, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                            branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], **kwargs, n_models=4)


def _100_3_ResNet18_byot_ALL(pretrained=False, **kwargs):
    return Three_ResNet_ALL(PreActBlock, num_classes=100, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                            branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], **kwargs, n_models=4)


def _10_1_ResNet18_byot_ALL(pretrained=False, **kwargs):
    return One_ResNet_ALL(PreActBlock, num_classes=10, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                          branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], **kwargs, n_models=4)


def _10_3_ResNet18_byot(n_models):  # (pretrained=False, **kwargs):
    return Three_ResNet(PreActBlock, num_classes=10, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                        branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], n_models=n_models)


def _100_3_ResNet18_byot(n_models):  # (pretrained=False, **kwargs):
    return Three_ResNet(PreActBlock, num_classes=100, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                        branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], n_models=n_models)


def _10_1_ResNet18_byot(n_models):  # (pretrained=False, **kwargs):
    return One_ResNet(PreActBlock, num_classes=10, num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                      branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], n_models=n_models)


def ResNet50_byot(pretrained=False, **kwargs):
    return Three_ResNet(Bottleneck, [3, 4, 6, 3], branch_layers=[[1, 1], [1]], **kwargs)

def shfl_resnet18(num_classes=10, model_idx=None):
    """
    SHFL 论文使用的 ResNet18 (基于 BYOT 结构)
    默认配置: 4 个分支 (Model-1 to Model-4)
    
    Args:
        num_classes: 分类数量
        model_idx: 模型深度索引 (1-4)，如果为None则返回完整模型
    """
    # 对应论文配置: Model-1~4
    # Block0 -> Exit0 (Model-1)
    # Block1 -> Exit1 (Model-2)
    # Block2 -> Exit2 (Model-3)
    # Block3 -> Exit3 (Model-4)
    
    # 这里的 branch_layers 参数对应的是 Bottleneck 的深度
    # 原始代码 _10_3_ResNet18_byot 用的是 [3, 2, 1, 0]
    # 我们直接复用它
    
    # 如果指定了model_idx，返回对应深度的模型
    if model_idx is not None:
        return _10_3_ResNet18_byot(n_models=model_idx)
    
    # 否则返回完整模型（用于Server端全局模型）
    return _10_3_ResNet18_byot(n_models=4) 


if __name__ == '__main__':
    net = CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    import sys

    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))