# 真实设备联邦学习框架

注意本仓库代码内编码了各个设备的ip和密码，在修改之前不要随意公开（12.28目前应该没有了）

运行方式为通过ssh连接各个设备和server，在server运行server开头的脚本，client运行client开头的相应脚本就行

脚本需要用sudo运行

> 注意要使用nohup或者tmux运行，因为设备进入休眠后会断开ssh连接

每个设备运行

```
sudo visudo
```

在文件最后加上

```
jetsonnano ALL=(ALL) NOPASSWD: /bin/systemctl suspend, /sbin/ethtool，/sbin/poweroff
```

用于不输入密码自动运行



client常用运行代码

```
nohup python3 client_heterofl.py --CID 0 > client_cid.log 2>&1 &
```



相应参数在utils/options.py

## 当前推荐入口：真实设备版 HELCFL

新的 baseline 入口是：

- `server_helcfl_real.py`
- `client_helcfl_real.py`

它们的设计原则是：

- 不使用 `wake/sleep`
- 继续复用现有 `ResNet18 + CIFAR10`
- 继续复用现有 socket 通信和 `hetero_model.resnet18(model_rate=...)`
- 使用 4 档模型宽度
- 使用 Jetson 原生 `nvpmodel` 两档自动变频

### 设备映射

当前默认：

- `CID 0-6 -> nano`
- `CID 7-8 -> agx_xavier`
- `CID 9 -> orinnanosuper`

这些映射定义在：

- `utils/helcfl_real_profiles.py`

### DVFS mode 配置

真实设备两档 mode 统一放在：

- `utils/helcfl_real_profiles.py`

当前默认先写成：

- `low = 0`
- `high = 1`

如果实验室里确认某种设备应改成别的 mode，只改这个文件，不改 server/client 主逻辑。

### sudo 权限

新的 client 会直接调用：

- `sudo nvpmodel -m <mode>`
- `sudo jetson_clocks`

因此需要配置免密 sudo。最少要保证运行用户对这两个命令无密码执行。

如果后续还要强制风扇，也建议一起放开对应 sysfs 写权限。

### 推荐运行方式

server：

```bash
nohup python3 server_helcfl_real.py > server_helcfl_real.log 2>&1 &
```

client：

```bash
nohup python3 client_helcfl_real.py --CID 0 > client_helcfl_real_0.log 2>&1 &
```

### 日志与能耗分析

最终能耗关系不是由代码直接计算，而是通过：

- server/client 日志时间戳
- 外部功率计 `xlsx`

来对齐分析。

日志时间格式统一为：

- `YYYY-MM-DD HH:MM:SS`

server 每轮会记录：

- 选中的客户端
- 模型宽度
- DVFS mode
- round 精度

client 每轮会记录：

- mode 切换开始/结束
- 训练开始/结束
- 上传开始/结束

这样后续可以直接和功率计的时间列对表。
