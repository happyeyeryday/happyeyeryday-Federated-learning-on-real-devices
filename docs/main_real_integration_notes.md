# Main Real Integration Notes

这份文档记录当前真实设备 `main_real` 集成的设计、代码落点、运行方式，以及它和 `HELCFL real` baseline 的区别。

## 1. 先回答两个直接问题

### 为什么复用 `client_helcfl_real.py`

因为这次主方法接入的变化点在 **server 决策器**，不在 **client 执行器**。

当前真实设备 client 做的事情本质上是：

1. 接收 server 下发的 `train_round` payload
2. 切换 DVFS
3. 按给定 `model_idx` 加载对应子模型
4. 本地训练
5. 上传权重和电量状态

这些动作对 `HELCFL real` 和 `main_real` 是一样的。  
主方法真正替换的是“server 每轮如何决定谁训练、训练哪个模型、用哪个 DVFS、谁 idle”。

所以：

- `client_helcfl_real.py` 不是“完全没区别”
- 而是“当前协议层面已经够用，不需要再复制一份新 client”
- 这次只额外加了 `--log_tag`，方便把 `main_real` 和 baseline 的 client 日志分开

如果后面你想把观测扩展到温度、带宽、内存压力、链路时延这些 **client 本地实时信号**，那时就要继续改 client 协议。

### `server_main_real.py` 能不能直接运行

能。

它就是新的 server 入口，直接运行方式是：

```bash
cd ~/server_fedavg
nohup python3 server_main_real.py \
  --policy_bundle ~/server_fedavg/policy_bundles/<bundle_dir> \
  --policy_mode offline_bundle \
  > server_main_real.log 2>&1 &
```

前提只有两个：

1. 环境里有仓库本来就依赖的包，比如 `loguru`、`ujson`
2. `--policy_bundle` 指向一个新格式 bundle，目录里至少有：
   - `low_level_eval_rnn.pt`
   - `high_level_role_eval.pt`
   - `policy_manifest.json`

所以不是“不能直接运行”，而是“不能空跑”，它需要离线训练导出的策略包。

## 2. 这次代码改了什么

### 仿真仓库

为了让真实设备能够严格消费主方法导出的策略包，我补了 manifest 字段：

- 文件：[agent.py](/mnt/sda/zzr/code_unzipped/recovered_from_zip/code/WITH/agent/agent.py)
- 新增内容：
  - `policy_manifest.json`
  - `config_path`
  - `seed`
  - `current_time`

### 真实设备仓库

#### 新增

- [server_main_real.py](/tmp/real_devices_master/server_main_real.py)
  - 新的真实设备主方法 server 入口
  - 基于 `SHFL_resnet` 聚合链路
  - 用离线 bundle 做每轮 server 侧推理

- [main_real_policy.py](/tmp/real_devices_master/utils/main_real_policy.py)
  - 负责加载 `policy_manifest.json`
  - 加载低层 `eval_rnn`
  - 加载高层 `role controller`
  - 维护 `last_actions / current_roles / role_age / hidden states`
  - 生成每轮 `action -> model_idx + dvfs + idle` 计划

- [deploy_main_real.md](/tmp/real_devices_master/docs/deploy_main_real.md)
  - 启动说明

#### 修改

- [options.py](/tmp/real_devices_master/utils/options.py)
  - 新增：
    - `--policy_bundle`
    - `--policy_manifest`
    - `--policy_mode`
    - `--log_tag`

- [client_helcfl_real.py](/tmp/real_devices_master/client_helcfl_real.py)
  - 增加 `--log_tag`
  - 不改训练协议

- [deploy_seed.py](/tmp/real_devices_master/deploy_sh/deploy_seed.py)
- [deploy_server.py](/tmp/real_devices_master/deploy_sh/deploy_server.py)
  - 把 `main_real` 新文件带进 deploy

## 3. `main_real` 和 baseline 的区别

这里 baseline 指当前真实设备主分支上的 `server_helcfl_real.py`。

### Baseline: `server_helcfl_real.py`

baseline 的核心是启发式规则：

- 用 `appearance_counter` 做简单公平性修正
- 用设备类型决定一个偏好的 `model_idx`
- 用手写规则决定 `dvfs_label`
- 每轮总会从活跃设备里选出一批训练设备

它的优点是简单、稳定、容易解释。  
它的缺点是规则是人工写死的，没有利用仿真里已经学到的联合策略。

### Main method: `server_main_real.py`

主方法的核心是离线学到的分层策略：

1. 高层 role controller 先决定设备当前属于哪类角色
   - `idle`
   - `train_low`
   - `train_high`
   - `train_any`
2. 低层 action RNN 再在对应动作子空间里选具体动作
   - 低 DVFS + 哪个模型
   - 高 DVFS + 哪个模型
   - 或者 idle

和 baseline 相比，主方法有几个本质区别：

- baseline 是“手工规则”
- main 是“离线训练得到的策略网络”

- baseline 的 `model_idx` 和 `dvfs` 基本是分开、按规则定
- main 是把 `参与/不参与`、`模型规模`、`DVFS 档位` 放在一个联合动作空间里选

- baseline 每轮一定按固定逻辑挑一批人训练
- main 可以显式让一部分设备 `idle`

- baseline 没有 RNN 隐状态
- main 会保留每个设备的历史隐状态，所以策略不是单轮独立决策

## 4. 为什么预期性能会更好

这里要说清楚：**这是“预期更好”，不是还没验证就能保证更好”。**

### 可能更好的原因

#### 1. 联合优化，而不是分开拍脑袋

主方法一次性决定：

- 这个设备这轮要不要参与
- 参与的话跑哪个模型
- 跑低 DVFS 还是高 DVFS

这比 baseline 的分步启发式更有机会找到更优组合。

#### 2. 允许主动 idle

baseline 更像“从可用设备里尽量选人干活”。  
主方法允许设备主动休眠一轮，这在电量受限时很关键：

- 能避免低价值训练把设备提前耗死
- 能把电量留给后面更有收益的轮次

如果目标是“直到设备没电前拿到更高精度”，这点通常很重要。

#### 3. 带历史的决策

主方法是 RNN + role controller，不是只看当前一个快照。  
所以它能隐式利用历史轨迹，例如：

- 最近是否频繁训练
- 最近是否一直在高功耗档
- 当前角色是否该切换

而 baseline 只有少量人工计数器。

#### 4. 训练目标和我们真实想要的目标更一致

主方法在仿真里学的是“精度、能量、时间、存活设备数”之间的折中。  
baseline 只是用人工 utility 去近似这个目标。

如果仿真到真实设备的落差可控，学到的策略通常会比手工规则更贴近真实优化目标。

## 5. 为什么现在还复用 baseline 的模型与通信链路

这是刻意做的，不是偷懒。

当前真实设备接入策略是：

- **不改 client 训练协议**
- **不改 SHFL_resnet 聚合链**
- **只替换 server 决策逻辑**

这样做的好处是：

- 风险小
- 更容易和 baseline 做公平对比
- 一旦结果有差异，更容易归因到“调度策略”而不是“训练链路变了”

所以现在的 `main_real` 可以理解成：

> 在 baseline 的真实设备执行框架上，替换掉它的启发式调度器。

## 6. 当前版本的边界

当前 `main_real` 是 v1，仍然有边界：

- 观测仍然是“电量比例 + 可行模型 mask”为主，和仿真一致优先
- 没把温度、带宽、链路抖动、内存压力这些实时信号接进策略
- idle 设备的能量下降是 server 侧估算，不是 client 实测
- 真正效果还要靠真实设备短程实验验证

所以更准确地说：

> 这不是“把所有真实设备因素都建模完了”，而是“先把主方法以最小侵入方式落到真实设备框架里”。

## 7. 推荐的对比方式

最合理的真实设备对比是：

1. 同样的 client 集合
2. 同样的初始电量
3. 同样的数据划分
4. 同样的 `SHFL_resnet` 模型与本地训练轮数
5. 只比较：
   - `server_helcfl_real.py`
   - `server_main_real.py`

这样才能把差异尽量归因到“主方法策略本身”。
