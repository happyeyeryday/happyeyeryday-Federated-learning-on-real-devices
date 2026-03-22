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