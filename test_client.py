"""
单模型能力测试脚本
测试不同 rate 的 HeteroFL 模型在 IID 数据上的准确率上限
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from loguru import logger
import time

from models.hetero_model import resnet18
from utils.get_dataset import get_dataset
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.FL_utils import Accumulator, accuracy


def train_single_model(model, trainloader, testloader, args, epochs=50):
    """
    训练单个模型并返回最终准确率
    """
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    best_acc = 0.0
    epoch_stats = []
    
    logger.info(f"{'='*60}")
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # 训练阶段
        for images, labels in trainloader:
            images, labels = images.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # 测试阶段（每 5 轮测试一次）
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            metric = Accumulator(2)
            
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(args.device), labels.to(args.device)
                    outputs = model(images)
                    metric.add(accuracy(outputs, labels), labels.numel())
            
            test_acc = metric[0] / metric[1] * 100
            best_acc = max(best_acc, test_acc)
            
            logger.info(f"Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Test Acc={test_acc:.2f}% (Best={best_acc:.2f}%)")
            epoch_stats.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': test_acc})
        else:
            logger.debug(f"Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}")
    
    return best_acc, epoch_stats


def test_rate_configurations(args):
    """
    测试不同 rate 的模型性能
    """
    # 加载数据集
    logger.info("Loading dataset...")
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    # 测试配置
    test_configs = [
        {'rate': 1.0, 'name': 'Full Model (Orin)'},
        {'rate': 0.75, 'name': '75% Model'},
        {'rate': 0.5, 'name': '50% Model (Xavier)'},
        {'rate': 0.25, 'name': '25% Model (Nano)'},
    ]
    
    results = []
    
    for config in test_configs:
        rate = config['rate']
        name = config['name']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 Testing: {name} (rate={rate})")
        logger.info(f"{'='*80}")
        
        # 创建模型
        model = resnet18(model_rate=rate, track=True)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Trainable parameters: {trainable_params:,}")
        logger.info(f"   - Model rate: {rate}")
        
        # 使用 Client 0 的数据（或全部数据）
        if args.use_single_client_data:
            # 使用单个客户端的数据（模拟联邦学习场景）
            client_idx = 0
            train_indices = dict_users[client_idx]
            train_subset = Subset(dataset_train, train_indices)
            trainloader = DataLoader(train_subset, batch_size=args.local_bs, shuffle=True)
            logger.info(f"   - Using Client {client_idx}'s data: {len(train_indices)} samples")
        else:
            # 使用全部训练数据（测试模型上限）
            trainloader = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)
            logger.info(f"   - Using full training data: {len(dataset_train)} samples")
        
        testloader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)
        
        # 训练
        start_time = time.time()
        best_acc, epoch_stats = train_single_model(
            model, 
            trainloader, 
            testloader, 
            args, 
            epochs=args.test_epochs
        )
        training_time = time.time() - start_time
        
        # 记录结果
        result = {
            'name': name,
            'rate': rate,
            'best_acc': best_acc,
            'total_params': total_params,
            'training_time': training_time,
            'epoch_stats': epoch_stats
        }
        results.append(result)
        
        logger.success(f"✅ {name}: Best Accuracy = {best_acc:.2f}%")
        logger.info(f"   Training time: {training_time:.2f}s")
    
    return results


def print_summary(results):
    """
    打印测试结果汇总
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 **FINAL RESULTS SUMMARY**")
    logger.info(f"{'='*80}")
    
    # 表格头
    logger.info(f"{'Model':<25} | {'Rate':<6} | {'Params':<12} | {'Best Acc':<10} | {'Time (s)':<10}")
    logger.info(f"{'-'*80}")
    
    # 表格内容
    for result in results:
        logger.info(
            f"{result['name']:<25} | "
            f"{result['rate']:<6.2f} | "
            f"{result['total_params']:>12,} | "
            f"{result['best_acc']:>9.2f}% | "
            f"{result['training_time']:>9.1f}"
        )
    
    logger.info(f"{'='*80}\n")
    
    # 分析
    logger.info("🔍 **Analysis:**")
    full_model_acc = next((r['best_acc'] for r in results if r['rate'] == 1.0), None)
    
    for result in results:
        if result['rate'] < 1.0 and full_model_acc:
            acc_ratio = result['best_acc'] / full_model_acc * 100
            param_ratio = result['rate'] * 100
            logger.info(
                f"   - {result['name']}: "
                f"{acc_ratio:.1f}% of full model accuracy with {param_ratio:.0f}% parameters"
            )


def save_results_to_file(results, args):
    """
    保存结果到文件
    """
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_model_test_{args.dataset}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.success(f"💾 Results saved to: {filename}")


if __name__ == '__main__':
    # 解析参数
    args = args_parser()
    
    # 添加测试专用参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epochs', type=int, default=50, 
                       help='Number of epochs to train each model')
    parser.add_argument('--use_single_client_data', action='store_true',
                       help='Use single client data instead of full dataset')
    test_args = parser.parse_args()
    
    # 合并参数
    args.test_epochs = test_args.test_epochs
    args.use_single_client_data = test_args.use_single_client_data
    
    # 设置设备
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) 
        if torch.cuda.is_available() and args.gpu != -1 
        else 'cpu'
    )
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 打印配置
    logger.info(f"{'='*80}")
    logger.info(f"🔧 Test Configuration:")
    logger.info(f"{'='*80}")
    logger.info(f"   - Dataset: {args.dataset}")
    logger.info(f"   - Device: {args.device}")
    logger.info(f"   - Test epochs: {args.test_epochs}")
    logger.info(f"   - Batch size: {args.local_bs}")
    logger.info(f"   - Learning rate: {args.lr}")
    logger.info(f"   - Momentum: {args.momentum}")
    logger.info(f"   - Data mode: {'Single Client' if args.use_single_client_data else 'Full Dataset'}")
    logger.info(f"   - Random seed: {args.seed}")
    logger.info(f"{'='*80}\n")
    
    # 运行测试
    results = test_rate_configurations(args)
    
    # 打印汇总
    print_summary(results)
    
    # 保存结果
    save_results_to_file(results, args)
    
    logger.success("🎉 All tests completed!")