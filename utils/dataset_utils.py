import os
import types
from collections import defaultdict

import ujson
import numpy as np
import json
import torch
import random


def check(config_path, train_path, test_path, num_clients, num_labels, niid=False,
        real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_labels'] == num_labels and \
            config['non_iid'] == niid and \
            config['real_world'] == real and \
            config['partition'] == partition:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def read_record(file):
    with open(file,"r") as f:
        dataJson = json.load(f)
        users_train = dataJson["train_data"]
        #users_test = dataJson["test_data"]
    dict_users_train = {}
    #dict_users_test = {}
    for key,value in users_train.items():
        newKey = int(key)
        dict_users_train[newKey] = value
    '''
    for key,value in users_test.items():
        newKey = int(key)
        dict_users_test[newKey] = value
    '''
    return dict_users_train #, dict_users_test

# def separate_data(train_data, num_clients, num_classes, beta=0.4):


#     y_train = np.array(train_data.targets)

#     min_size_train = 0
#     min_require_size = 10
#     K = num_classes

#     N_train = len(y_train)
#     dict_users_train = {}

#     while min_size_train < min_require_size:
#         idx_batch_train = [[] for _ in range(num_clients)]
#         idx_batch_test = [[] for _ in range(num_clients)]
#         for k in range(K):
#             idx_k_train = np.where(y_train == k)[0]
#             np.random.shuffle(idx_k_train)
#             proportions = np.random.dirichlet(np.repeat(beta, num_clients))
#             proportions_train = np.array([p * (len(idx_j) < N_train / num_clients) for p, idx_j in zip(proportions, idx_batch_train)])
#             proportions_train = proportions_train / proportions_train.sum()
#             proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
#             idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
#             min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
#             # if K == 2 and n_parties <= 10:
#             #     if np.min(proportions) < 200:
#             #         min_size = 0
#             #         break

#     for j in range(num_clients):
#         np.random.shuffle(idx_batch_train[j])
#         dict_users_train[j] = idx_batch_train[j]

#     train_cls_counts = record_net_data_stats(y_train,dict_users_train)
#     # print("=" * 20 + " 数据分布检查 " + "=" * 20)
#     # for client_id, data_idxs in dict_users_train.items():
#     #     print(f"Client {client_id}: {len(data_idxs)} samples")
#     # print("=" * 60)
#     return dict_users_train
def separate_data(train_data, num_clients, num_classes, beta=0.4):
    y_train = np.array(train_data.targets)
    N_train = len(y_train)
    K = num_classes
    
    # 1. 定义目标配额 (Quota)
    target_quotas = np.zeros(num_clients, dtype=int)
    nano_ids = list(range(0, 7))       # 0-6
    powerful_ids = list(range(7, 10))  # 7, 8, 9
    
    target_quotas[nano_ids] = 2000
    remaining_data = N_train - sum(target_quotas[nano_ids])
    # Xavier 和 Orin 平分剩下的 36000
    avg_powerful = remaining_data // len(powerful_ids)
    target_quotas[powerful_ids] = avg_powerful
    target_quotas[9] += (remaining_data % len(powerful_ids)) # 余数给 Orin
    MIN_CLASSES = 3
    
    MAX_RETRIES = 50
    retry_count = 0
    
    dict_users_train = {}

    while retry_count < MAX_RETRIES:
        dict_users_train = {i: [] for i in range(num_clients)}
        current_counts = np.zeros(num_clients, dtype=int)
        
        # 3. 按类别进行分配
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            
            # 生成 Dirichlet 比例
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            
            # 软约束：根据剩余容量调整比例
            mask = np.ones(num_clients)
            for i in range(num_clients):
                capacity = target_quotas[i] - current_counts[i]
                if capacity <= 0:
                    mask[i] = 0
                elif current_counts[i] / target_quotas[i] > 0.95:
                    mask[i] = 0.1 # 接近满时，降低获取新数据的概率
            
            proportions = proportions * mask
            if proportions.sum() > 0:
                proportions /= proportions.sum()
            else:
                # 兜底：如果大家都快满了，强行给还有空位的设备
                rem_mask = (current_counts < target_quotas).astype(float)
                proportions = rem_mask / (rem_mask.sum() + 1e-6)

            # 计算各设备分多少
            count_k = len(idx_k)
            actual_counts_k = (proportions * count_k).astype(int)
            
            # 确保 Nano 绝不越界
            for i in nano_ids:
                rem = target_quotas[i] - current_counts[i]
                if actual_counts_k[i] > rem:
                    actual_counts_k[i] = rem
            
            # 处理溢出：将剩下的数据给强大的设备
            allocated = sum(actual_counts_k)
            surplus = count_k - allocated
            if surplus > 0:
                for p_id in powerful_ids:
                    p_rem = target_quotas[p_id] - current_counts[p_id] - actual_counts_k[p_id]
                    if p_rem > 0:
                        add_num = min(surplus, p_rem)
                        actual_counts_k[p_id] += add_num
                        surplus -= add_num
                        if surplus <= 0: break
            
            # 实际执行切分并存入字典
            split_points = np.cumsum(actual_counts_k)[:-1]
            for i, idx_list in enumerate(np.split(idx_k, split_points)):
                dict_users_train[i].extend(idx_list.tolist())
                current_counts[i] += len(idx_list)

        # 验证类别数量
        min_classes_found = min([len(np.unique(y_train[dict_users_train[i]])) for i in range(num_clients)])
        
        if min_classes_found >= MIN_CLASSES:
            break
        else:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print(f"⚠️ 警告: 达到最大重试次数，某些设备类别数可能 < {MIN_CLASSES}")
                break

    # 4. [关键修复] 按类别平衡地调整配额，保持non-IID分布
    for i in nano_ids:
        current_size = len(dict_users_train[i])
        diff = target_quotas[i] - current_size
        
        if diff > 0:  # Nano 需要补充数据
            # 按类别比例从powerful设备"借"数据
            needed_per_class = {}
            for cls in range(K):
                cls_count = sum([1 for idx in dict_users_train[i] if y_train[idx] == cls])
                needed_per_class[cls] = max(1, int(diff * cls_count / max(current_size, 1)))
            
            for cls, needed in needed_per_class.items():
                transferred = 0
                for donor in powerful_ids:
                    if transferred >= needed:
                        break
                    # 从donor中找属于cls的样本
                    donor_cls_indices = [idx for idx in dict_users_train[donor] if y_train[idx] == cls]
                    transfer_count = min(needed - transferred, len(donor_cls_indices), len(dict_users_train[donor]) - target_quotas[donor] + 100)
                    
                    for _ in range(transfer_count):
                        if donor_cls_indices:
                            sample_idx = donor_cls_indices.pop()
                            dict_users_train[donor].remove(sample_idx)
                            dict_users_train[i].append(sample_idx)
                            transferred += 1
                
                if len(dict_users_train[i]) >= target_quotas[i]:
                    break
        
        elif diff < 0:  # Nano 数据过多（罕见）
            # 转移多余数据给powerful设备
            excess = -diff
            recipient = powerful_ids[np.argmin([len(dict_users_train[p]) for p in powerful_ids])]
            for _ in range(min(excess, len(dict_users_train[i]) - target_quotas[i])):
                if dict_users_train[i]:
                    dict_users_train[recipient].append(dict_users_train[i].pop())

    # 5. 打乱每个客户端的数据（保证可复现性）
    for i in range(num_clients):
        np.random.shuffle(dict_users_train[i])
    
    # 6. 打印审计报告（包含类别分布统计）
    print("\n" + "═" * 25 + " 数据分配最终审计 " + "═" * 25)
    total_samples = 0
    for i in range(num_clients):
        dtype = "Nano" if i in nano_ids else ("Xavier" if i in [7, 8] else "Orin")
        client_indices = dict_users_train[i]
        num_cls = len(np.unique(y_train[client_indices]))
        
        # 计算类别分布的不均衡度（标准差）
        cls_counts = [sum([1 for idx in client_indices if y_train[idx] == c]) for c in range(K)]
        cls_std = np.std(cls_counts)
        
        status = "✅" if num_cls >= MIN_CLASSES else "❌"
        total_samples += len(client_indices)
        print(f"{status} ID {i:d} ({dtype:6s}) | 样本: {len(client_indices):5d} | 类别: {num_cls:2d}/{K} | 不均衡度: {cls_std:.1f}")
    
    print(f"Total samples: {total_samples}/{N_train}")
    print("═" * 66 + "\n")
    
    # [可选] 调用详细统计
    # record_net_data_stats(y_train, dict_users_train)
    
    return dict_users_train

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():

        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))

    return net_cls_counts

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                num_labels, statistic, niid=False, real=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_labels': num_labels,
        'non_iid': niid,
        'real_world': real,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
    }

    # gc.collect()

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(train_dict, f)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(test_dict, f)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list

def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = {i: [] for i in range(num_users)}
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

def gen_random_loaders(dataset, num_users, rand_set_all = None, classes_per_user=2):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    if rand_set_all is None:
        rand_set_all = gen_classes_per_node(dataset, num_users, classes_per_user)

    usr_subset_idx = gen_data_split(dataset, num_users, rand_set_all)

    #cls_counts = record_net_data_stats(dataset.targets, usr_subset_idx)

    return usr_subset_idx,rand_set_all