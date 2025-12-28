import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import wandb

dates = pd.read_csv('../result/output/5/0.1/cifar10_SplitFed_resnet_test_acc_2000_lr_0.005_normal_2_2023_10_17_12_21_23_client2.txt', header=None, sep=" ")
normal_noiid1_client2 = np.array(dates[dates.columns[1:2001]]).tolist()[0]
dates = pd.read_csv('../result/output/5/0.1/cifar10_SplitFed_resnet_test_acc_2000_lr_0.005_balance_2_2023_10_17_11_50_01_client2.txt', header=None, sep=" ")
balance_noiid1_client2 = np.array(dates[dates.columns[1:2001]]).tolist()[0]
dates = pd.read_csv('../result/output/5/0.1/cifar10_HetroSplitFed_resnet_test_acc_2000_lr_0.005_normal_2_2023_10_18_12_26_22.txt', header=None, sep=" ")
hetro_normal_noiid1 = np.array(dates[dates.columns[1:2001]]).tolist()[0]
dates = pd.read_csv('../result/output/5/0.1/cifar10_HetroSplitFed_resnet_test_acc_2000_lr_0.005_balance_2_2023_10_18_11_53_45.txt', header=None, sep=" ")
hetro_balance_noiid1 = np.array(dates[dates.columns[1:2001]]).tolist()[0]


wandb.init(
    # set the wandb project where this run will be logged
    project="Hetro split fed",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.005,
        "architecture": "ResNet18",
        "dataset": "CIFAR-10 && VGG",
        "epochs": 2000,
    }
)

# for epoch in range(1000):
#     wandb.log({"normal_noiid0.5_test_acc": normal_noiid5_testacc[epoch],
#                "normal_noiid1.0_test_acc": normal_noiid10_testacc[epoch],
#                "normal_noiid0.1_test_acc": normal_noiid1_testacc[epoch],
#                "balance2_noiid1.0_test_acc": balance2_noiid10_testacc[epoch],
#                "balance2_noiid0.5_test_acc": balance2_noiid5_testacc[epoch],
#                "balance2_noiid0.1_test_acc": balance2_noiid1_testacc[epoch],
#                "balance4_noiid1.0_test_acc": balance4_noiid10_testacc[epoch],
#                "balance4_noiid0.5_test_acc": balance4_noiid5_testacc[epoch],
#                "balance4_noiid0.1_test_acc": balance4_noiid1_testacc[epoch],
#                "marry4_noiid1.0_test_acc": marry4_noiid10_testacc[epoch],
#                "marry4_noiid0.5_test_acc": marry4_noiid5_testacc[epoch],
#                "marry4_noiid0.1_test_acc": marry4_noiid1_testacc[epoch],
#                "cross4_noiid1.0_test_acc": cross4_noiid10_testacc[epoch],
#                "cross4_noiid0.5_test_acc": cross4_noiid5_testacc[epoch],
#                "cross4_agg5_noiid1.0_test_acc": cross4_agg5_noiid10_testacc[epoch],
#                "cross4_agg5_noiid0.5_test_acc": cross4_agg5_noiid5_testacc[epoch],
#                "cross4_agg5_noiid0.1_test_acc": cross4_agg5_noiid1_testacc[epoch],
#                "avg_server2_noiid1.0_test_acc": avg_server2_noiid10_testacc[epoch],
#                "avg_server2_noiid0.5_test_acc": avg_server2_noiid5_testacc[epoch],
#                "avg_server2_noiid0.1_test_acc": avg_server2_noiid1_testacc[epoch],
#                "avg_server2_agg5_noiid1.0_test_acc": avg_server2_agg5_noiid10_testacc[epoch],
#                "avg_server2_agg5_noiid0.5_test_acc": avg_server2_agg5_noiid5_testacc[epoch],
#                "avg_server2_agg5_noiid0.1_test_acc": avg_server2_agg5_noiid1_testacc[epoch],
#                "avg_server4_noiid1.0_test_acc": avg_server4_noiid10_testacc[epoch],
#                "avg_server4_noiid0.5_test_acc": avg_server4_noiid5_testacc[epoch],
#                "avg_server4_noiid0.1_test_acc": avg_server4_noiid1_testacc[epoch],
#                "balance_server2_agg5_noiid1.0_test_acc": balance_server_noiid10_testacc[epoch]
#                })
# "normal_noiid10_client2": normal_noiid10_client2[epoch],
#                "normal_noiid10_client3": normal_noiid10_client3[epoch],
#                "normal_noiid10_client4": normal_noiid10_client4[epoch],
#                "balance_noiid10_client2": balance_noiid10_client2[epoch],
#                "balance_noiid10_client3": balance_noiid10_client3[epoch],
#                "balance_noiid10_client4": balance_noiid10_client4[epoch],
#                "hetro_normal_noiid10": hetro_normal_noiid10[epoch],
#                "hetro_balance_noiid10": hetro_balance_noiid10[epoch],
#                "normal_noiid5_client2": normal_noiid5_client2[epoch],
#                "normal_noiid5_client3": normal_noiid5_client3[epoch],
#                "normal_noiid5_client4": normal_noiid5_client4[epoch],
#                "balance_noiid5_client2": balance_noiid5_client2[epoch],
#                "balance_noiid5_client3": balance_noiid5_client3[epoch],
#                "balance_noiid5_client4": balance_noiid5_client4[epoch],
#                "hetro_normal_noiid5": hetro_normal_noiid5[epoch],
#                "hetro_balance_noiid5": hetro_balance_noiid5[epoch],

for epoch in range(2000):
    wandb.log({"normal_noiid1_client2": normal_noiid1_client2[epoch],
               "balance_noiid1_client2": balance_noiid1_client2[epoch],
               "hetro_normal_noiid1": hetro_normal_noiid1[epoch],
               "hetro_balance_noiid1": hetro_balance_noiid1[epoch]
               })


