import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用原来的基础块
from models.SHFL_resnet import conv3x3, PreActBlock

class Three_ResNet_Distill(nn.Module):
    """
    专门用于自蒸馏的 Client 模型。
    区别：forward 返回 [exit0_out, exit1_out, ..., final_out]
    而不是只返回 final_out
    """
    def __init__(self, block, num_classes, branch_layers=[], planes=[], strides=[], num_blocks=[], is_bias=True,
                 n_models=4):
        super(Three_ResNet_Distill, self).__init__()
        self.branch_layers = branch_layers
        
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        
        mainblocks = []
        bottlenecks = []
        fcs = []
        
        self.n_models = n_models
        # 注意：这里的 n_models 实际上是当前 Client 被分配的模型深度 (1-4)
        
        # 构建主干和出口
        temp_in_planes = 64
        for i in range(n_models):
            self.in_planes = temp_in_planes
            mainblocks.append(self._make_resblocklayer(block, planes[i], num_blocks[i], strides[i]))
            temp_in_planes = self.in_planes

        bottleneck_input_dims = [64, 128, 256, 512]
        for i in range(n_models):
            self.in_planes = bottleneck_input_dims[i]
            # 每个出口都加上，为了自蒸馏
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

    def _make_bottlenecklayer(self, block, num_classes, branch_layers, stride):
        layers = []
        input_planes = self.in_planes
        for i in range(branch_layers):
            layers.append(block(input_planes, input_planes * 2, stride))
            input_planes = input_planes * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 逐层前向传播，并在每个出口收集结果
        for idx in range(self.n_models):
            out = self.mainblocks[idx](out)
            
            # 计算该层的出口输出
            features_end = self.bottlenecks[idx](out)
            pooled = self.avgpool(features_end)
            flattened = pooled.view(pooled.size(0), -1)
            logits = self.fcs[idx](flattened)
            
            outputs.append(logits)
            
        return outputs # 返回列表

def shfl_resnet18_distill(num_classes=10, model_idx=None):
    """工厂函数：返回支持自蒸馏的模型"""
    # 参数配置复用 _10_3_ResNet18_byot 的配置
    return Three_ResNet_Distill(PreActBlock, num_classes=num_classes, 
                                num_blocks=[2, 2, 2, 2], planes=[64, 128, 256, 512],
                                branch_layers=[3, 2, 1, 0], strides=[1, 2, 2, 2], 
                                n_models=model_idx if model_idx else 4)