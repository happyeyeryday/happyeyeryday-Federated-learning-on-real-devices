import time

import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import random
import numpy as np

from utils.ConnectHandler_server import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
from models.SplitModel import ResNet18_client_side, ResNet18_server_side, VGG16_client_side, VGG16_server_side, \
    ResNet8_client_side, ResNet8_server_side, ResNet18_entire, VGG16_entire, ResNet8_entire
import copy
import multiprocessing
import threading
from loguru import logger

net_glob = None
Reuse_ratio = 1
num_device = 10
w_local_client = []


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def FedAvg(w):
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


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    # if args.algorithm in ['FedAvg']:
    if 'resnet18' in args.model:
            net_glob = ResNet18_entire()
            net_glob.apply(init_weights)
            net_glob.to(args.device)
        # if 'vgg' in args.model:
        #     net_glob = VGG16_entire()
        #     net_glob.apply(init_weights)
        #     net_glob.to(args.device)
        # if 'resnet8' in args.model:
        #     net_glob = ResNet8_entire()
        #     net_glob.apply(init_weights)
        #     net_glob.to(args.device)
    summary_acc_test_collect = []
    time_collect = []
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))


    num_device = int(args.num_users / Reuse_ratio)
    m = max(int(args.frac * num_device), 1)

    # connectHandler = ConnectHandler(args.num_users, args.HOST, args.POST)
    connectHandler = ConnectHandler(num_device, args.HOST, args.POST)
    for iter in range(args.epochs):
        idxs_devices = np.random.choice(range(num_device), m, replace=False)
        idxs_drift = np.random.randint(0, Reuse_ratio, size=len(idxs_devices))
        idxs_users = idxs_devices + num_device * idxs_drift
        print("round:", iter, " choose client:",idxs_users)
        w_local_client = []

        for idx in idxs_users:
            msg = dict()
            msg['net'] = copy.deepcopy(net_glob)
            msg['idxs_list'] = dict_users[idx]
            msg['type'] = 'net'
            msg['round'] = iter
            logger.info("send net to client {}".format(idx))
            connectHandler.sendData(idx % num_device, msg)

        while len(w_local_client) < len(idxs_users):
            print("w_local_client:", len(w_local_client))
            msg, client_idx = connectHandler.receiveData()
            print("recv net from client {}".format(client_idx))
            if msg['type'] == 'net':
                net = msg['net']
                w_local_client.append(net)

        w_local = [copy.deepcopy(net.state_dict()) for net in w_local_client]
        w_glob = FedAvg(w_local)
        net_glob.load_state_dict(w_glob)

        acc = summary_evaluate(copy.deepcopy(net_glob).to(args.device),
                               dataset_test, args.device) * 100
        current_time = time.time()
        summary_acc_test_collect.append(acc)
        time_collect.append(current_time)

        print("====================== SERVER V1==========================")
        print(' Test: Round {:3d}, Current time {:.3f}, Avg Accuracy {:.3f}'.format(iter, current_time, acc))
        print("==========================================================")

    save_result(summary_acc_test_collect, 'test_acc', args)
    save_result(time_collect, 'time', args)