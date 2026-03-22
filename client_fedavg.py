import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
from pandas import DataFrame
import random
import numpy as np
import matplotlib

from utils.ConnectHandler_client import ConnectHandler
from utils.FL_utils import *
from utils.get_dataset import *
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.utils import save_result
import copy
from loguru import logger

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    ID = args.CID
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    local_net = None
    dtLoader = None
    my_train_dict = None
    optimizer = None
    loss_func = nn.CrossEntropyLoss()

    while True:
        recv = connectHandler.receiveFromServer()
        if recv['type'] == 'net':
            local_net = recv['net']
            idxs_list = recv['idxs_list']
            round = recv["round"]
            dtLoader = DataLoader(DatasetSplit(dataset_train, idxs_list),
                                  batch_size=args.bs, shuffle=True)
            local_net.train()
            optimizer = torch.optim.SGD(local_net.parameters(), lr=args.lr * (args.lr_decay ** round),
                                        momentum=args.momentum, weight_decay=args.weight_decay)

            for batch_idx, (images, labels) in enumerate(dtLoader):
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                fx = local_net(images)['output']
                loss = loss_func(fx, labels)
                loss.backward()
                optimizer.step()

            msg = dict()
            msg['type'] = 'net'
            msg['net'] = copy.deepcopy(local_net)
            connectHandler.uploadToServer(msg)
