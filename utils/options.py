#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.998, help="learning rate decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay (default: 1e-4)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="heterofl")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=1, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=5, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=0.1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # FedProx
    parser.add_argument('--prox_alpha', type=float, default=0.01, help='The hypter parameter for the FedProx')
    # SCAFFOLD
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    # Moon
    parser.add_argument('--contrastive_alpha', type=float, default=5, help='The hypter parameter for the Moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    # FedGKD
    parser.add_argument('--ensemble_alpha', type=float, default=0.2, help='The hypter parameter for the FedGKD')
    # FedDC
    parser.add_argument('--sim_type', type=str, default='L1', help='Cluster Sampling: cosine or L1 or L2')
    # FedDC
    parser.add_argument('--alpha_coef', type=float, default=1e-2, help='FedDC')
    # FedMLB
    parser.add_argument("--temp",default=1,type=float,metavar="N",help="temperature")
    parser.add_argument("--lambda1",default=1,type=float,metavar="N",help="Weight for CE loss of main pathway")
    parser.add_argument("--lambda2",default=1,type=float,metavar="N",help="Weight for CE loss of hybrid pathways")
    parser.add_argument("--lambda3",default=1,type=float,metavar="N",help="Weight for KD loss of hybrid pathways")

    # SplitFed && HetroSplitFed
    parser.add_argument("--divide_strategy", default="normal", type=str)
    parser.add_argument("--model_chosen", default="adaptive", type=str)
    parser.add_argument("--SFL_type", default="V2", type=str)
    parser.add_argument("--group_client_cnt", default=1, type=int)
    parser.add_argument("--batch_len", default=1000, type=int)
    parser.add_argument("--server_net_cnt", default=4, type=int)
    parser.add_argument("--agg_period", default=1, type=int)

# ================= ScaleFL / HeteroFL 新增参数 =================
    # 1. 本地训练轮数 (修复 Bug 1)
    # parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs: E")

    # 2. BN层统计量追踪 (修复 Bug 2)
    # ScaleFL 推荐 False (即使用 sBN)，设为 0 表示 False
    parser.add_argument('--track_running_stats', type=int, default=0, help='0 for False (sBN), 1 for True')

    # 3. 自蒸馏 (Self-Distillation) 参数
    parser.add_argument('--KD_T', type=float, default=4.0, help='temperature for KD')
    parser.add_argument('--KD_gamma', type=float, default=0.5, help='weight for KD loss')
    # ==============================================================

    # connect config
    parser.add_argument("--HOST", default='192.168.31.105', type=str)
    parser.add_argument("--POST", default=8080, type=int)
    parser.add_argument("--CID", default=0, type=int)
    parser.add_argument("--policy_bundle", default='', type=str)
    parser.add_argument("--policy_manifest", default='', type=str)
    parser.add_argument("--policy_mode", default='offline_bundle', type=str)
    parser.add_argument("--log_tag", default='', type=str)
    args = parser.parse_args()
    return args
