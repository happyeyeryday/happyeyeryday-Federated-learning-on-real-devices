import os
import time
import subprocess
from loguru import logger
from ping3 import ping
# ==========================================
# 功耗与电池配置区域
# ==========================================

POWER_CONFIG = {
    'orin': {
        'train': 20.6937,   # 训练时的平均功耗
        'idle': 7.6252,     # 开机但未训练时的功耗 (等待)
        'sleep': 1.5524,    # suspend 状态下的功耗
        'communication':7.723,
    },
    'xavier': {
        'train': 16.0971,
        'idle': 5.7406,
        'sleep': 1.8653,
        'communication':5.7716,

    },
    'nano': {
        'train': 10.013,
        'idle': 2.9454,
        'sleep': 0.4328,
        'communication':3.0533,
    }
}

# [TODO] 在此处填写你为设备模拟的总电量 (单位：焦耳)
BATTERY_CAPACITY = {
    'orin': 50000,  # J
    'xavier': 50000, # J
    'nano': 50000,   # J
}


class BatteryManager:
    """
    负责模拟和管理客户端设备电池电量的类。
    """
    def __init__(self, device_type: str):
        if device_type not in POWER_CONFIG:
            raise ValueError(f"Unknown device_type: {device_type}. Must be one of {list(POWER_CONFIG.keys())}")
        
        self.device_type = device_type
        self.power_profile = POWER_CONFIG[device_type]
        self.total_capacity = BATTERY_CAPACITY[device_type]
        self.current_charge = float(self.total_capacity) # 当前电量 (J)

        logger.info(f"🔋 [Battery] Initialized. Capacity: {self.current_charge} J")

    def consume(self, activity: str, duration_seconds: float):
        """
        根据活动和持续时间计算并扣除电量。

        Args:
            activity (str): 活动类型 ('train', 'idle', 'sleep')。
            duration_seconds (float): 活动持续的时间（秒）。
        """
        if activity not in self.power_profile:
            logger.warning(f"Unknown activity '{activity}'. Using 'idle' power consumption.")
            activity = 'idle'
            
        power_w = self.power_profile[activity]
        
        # 功耗 (J) = 功率 (W) * 秒(s)
        consumed_joules = power_w * duration_seconds
        self.current_charge -= consumed_joules
        
        # 确保电量不为负
        self.current_charge = max(0, self.current_charge)
        
        percentage = (self.current_charge / self.total_capacity) * 100
        
        logger.info(
            f"⚡ [Battery] Activity: {activity} ({duration_seconds:.1f}s). "
            f"Consumed: {consumed_joules:.4f} J. "
            f"Remaining: {self.current_charge:.2f}/{self.total_capacity} J ({percentage:.1f}%)"
        )
        return self.current_charge

    def check_energy(self, threshold=50.0):
        """
        检查当前电量是否高于阈值
        """
        is_enough = self.current_charge > threshold
        if not is_enough:
            logger.warning(f"🪫 [Battery] Low Level! Current: {self.current_charge:.1f} J < Threshold: {threshold} J")
        return is_enough

# ==========================================
# Client 端使用的功能
# ==========================================

def smart_sleep(server_ip, interface="eth0", ping_timeout=15):
    """
    Client 端智能休眠函数。
    适配 Orin/Xavier/Nano，解决网卡驱动导致的死机/无法唤醒问题。
    
    Args:
        server_ip (str): Server 的 IP 地址，用于唤醒后检测网络恢复。
        interface (str): 网卡接口名称，默认为 eth0。
        ping_timeout (int): 等待网络恢复的最长超时时间(秒)。
    """
    logger.info(f"💤 [PowerManager] 任务完成，准备进入休眠 (Interface: {interface})...")

    # 1. 开启 WoL (Orin 可能会在此处瞬间断网，分开执行是关键)
    # 这一步是为了确保下次能被唤醒
    exit_code = os.system(f"sudo ethtool -s {interface} wol g")
    if exit_code != 0:
        logger.warning("⚠️ 设置 WoL 失败，请检查 sudo 免密权限！")

    # 2. 关键停顿：给驱动层缓冲时间，防止时序竞争导致系统死锁
    time.sleep(2)

    # 3. 执行系统挂起
    # 程序会阻塞在这里，直到被物理唤醒
    os.system("sudo systemctl suspend")

    time.sleep(2)

    # ========================================
    # ⚡ 硬件在此处停止运行
    # ⚡ 收到 Server 的 Magic Packet
    # ☀️ 硬件被唤醒，代码继续执行
    # ========================================

    logger.info("☀️ [PowerManager] 设备已唤醒！正在检测网络连通性...")
    _wait_for_network(server_ip, timeout=ping_timeout)


def _wait_for_network(server_ip, timeout):
    """(内部函数) 循环 Ping Server 直到网络恢复"""
    start_time = time.time()
    retry_count = 0
    
    while True:
        # 检查是否超时
        if time.time() - start_time > timeout:
            logger.error(f"❌ [PowerManager] 网络恢复超时 ({timeout}s)！继续尝试运行...")
            break

        # Ping Server (超时1秒)
        # > /dev/null 2>&1 用于屏蔽 ping 的命令行输出
        ret = os.system(f"ping -c 1 -W 1 {server_ip} > /dev/null 2>&1")
        
        if ret == 0:
            duration = int(time.time() - start_time)
            logger.info(f"✅ [PowerManager] 网络已恢复！(耗时 {duration} 秒)")
            break
        else:
            retry_count += 1
            if retry_count % 5 == 0:
                logger.info(f"⏳ [PowerManager] 等待网络... (已尝试 {retry_count} 次)")
            time.sleep(1)


# ==========================================
# Server 端使用的功能
# ==========================================

def wake_clients(mac_to_ip_map, total_timeout):
    """
    Server 端批量唤醒函数 (主动确认与重试版)。
    
    Args:
        mac_to_ip_map (dict): 字典，{MAC地址: IP地址}
        total_timeout (int): 整个唤醒过程的最大超时时间(秒)。
    """
    if not mac_to_ip_map:
        return

    # 初始化待唤醒列表
    devices_to_wake = list(mac_to_ip_map.keys())
    
    logger.info(f"⚡ [PowerManager] 开始唤醒 {len(devices_to_wake)} 台设备...")
    
    # 1. 初始批量唤醒
    for mac in devices_to_wake:
        # 发送 3 个包以提高成功率
        for i in range(0,3):
            cmd = f"wakeonlan {mac} > /dev/null 2>&1"
            os.system(cmd)
            time.sleep(1)
    
    logger.info("初始唤醒包已发送。开始轮询检查设备状态...")

    start_time = time.time()
    
    # 2. 循环检查与重试，直到所有设备都上线或超时
    while devices_to_wake and (time.time() - start_time) < total_timeout:
        
        # 遍历所有还未唤醒的设备
        # 使用 [:] 创建副本，以便在循环中安全地修改列表
        for mac in devices_to_wake[:]:
            ip = mac_to_ip_map[mac]
            
            # 使用 ping3 检查，超时设为 1 秒
            is_alive = ping(ip, timeout=1)
            
            if is_alive is not False and is_alive is not None:
                # 如果 ping 通了 (返回延迟时间)
                logger.success(f"✅ 设备 {ip} ({mac}) 已上线！")
                devices_to_wake.remove(mac)
            else:
                # 如果没 ping 通
                logger.warning(f"⏳ 设备 {ip} ({mac}) 仍未响应，重新发送唤醒包...")
                os.system(f"wakeonlan {mac} > /dev/null 2>&1")
        
        # 如果还有设备没醒，等待几秒再进行下一轮检查
        if devices_to_wake:
            remaining_time = int(total_timeout - (time.time() - start_time))
            logger.info(f"还有 {len(devices_to_wake)} 台设备未唤醒。将在 5 秒后重试... (剩余 {remaining_time}s)")
            time.sleep(5)
    
    # 3. 最终结果
    if not devices_to_wake:
        logger.success("🎉 [PowerManager] 所有设备已成功唤醒！")
    else:
        logger.error(f"❌ [PowerManager] 唤醒超时！以下设备未能唤醒:")
        for mac in devices_to_wake:
            logger.error(f"  - IP: {mac_to_ip_map[mac]}, MAC: {mac}")