# Main Real Change Summary

这份文档只说明这次“真实 Jetson 上复现主方法”具体改了什么，不展开部署细节。

## 1. 这次新增了什么

主方法现在是一对独立脚本：

- `server_main_real.py`
- `client_main_real.py`

这对脚本和 baseline 分开：

- baseline 继续是 `server_helcfl_real.py + client_helcfl_real.py`
- 主方法现在是 `server_main_real.py + client_main_real.py`

这样做的目的很明确：

- 不污染已经跑通的 baseline
- 主方法的 server/client 训练语义保持一致
- 后续真实设备对比时，边界清楚

## 2. Server 做了什么

`server_main_real.py` 现在做的是两部分事情：

### 决策

- 加载离线训练好的策略包
- 用 `MainRealPolicy` 给每个设备决定：
  - 这轮是否参与
  - 参与的话用哪个 `model_idx`
  - 使用哪个 `dvfs_mode`

### 联邦训练

- 全局模型改成 `SHFL_resnet` 的 BYOT 版本
- 下发给 client 的不再是单出口切片，而是“到当前 `model_idx` 为止”的多出口切片
- 聚合时直接按 `mainblocks / bottlenecks / fcs` 的同名参数聚合
- 每轮评估记录 4 个出口的精度，主指标仍然默认看最后一个出口

## 3. Client 做了什么

`client_main_real.py` 不再复用 `client_helcfl_real.py` 的单出口训练。

它现在的本地训练语义是：

- 模型：`shfl_resnet18_distill`
- 前向输出：`[exit1, ..., final]`
- loss：
  - 单出口时：普通 CE
  - 多出口时：中间出口对最终出口蒸馏

也就是说，真实设备主方法现在和仿真里的 “BYOT/self-distill + 分层调度” 语义是对齐的，不再是“server 是主方法，client 其实还是 baseline 训练”。

## 4. 这次复用了哪些接口

为了尽量不推翻已有真实设备代码，这次保留了这些接口不变：

- `ConnectHandler_server / ConnectHandler_client`
- `train_round -> client_update -> upload_ack`
- `args_parser`
- `dvfs_mode`
- `battery_joules / battery_level`
- 低电量关机与 ACK 流程

所以这次是“换主方法逻辑”，不是“重做通信框架”。

## 5. 新增的共享模块

为了让主方法链独立且清晰，这次还新增了两个共享模块：

- `utils/main_real_profiles.py`
  - 管理主方法使用的 `model_idx / depth_ratio / DVFS` 映射

- `utils/power_manager_real.py`
  - 管理真实设备电量、模式、电量消耗估算

## 6. 当前状态

已经完成的：

- 新 server/client 代码已落地
- 语法检查通过
- 导入检查通过
- 结构检查通过：
  - server 可以按 `model_idx` 切出 BYOT 多出口权重
  - client 的 distill 模型可以加载并前向
  - 聚合函数可以接收回传参数

还没做的：

- 没做真实 Jetson 端到端联调
- 没做长程真实设备实验

所以当前最准确的说法是：

> 代码链路已经补齐，具备上机条件；但还没有做真实设备运行验证。
