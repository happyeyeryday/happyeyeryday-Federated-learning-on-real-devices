import paramiko
from scp import SCPClient
import os

# ================= 配置区域 =================

# 1. 要传输的文件或文件夹路径列表 (Windows 路径请在前面加 r，或使用双反斜杠)
# 支持文件夹（会自动递归传输）和单个文件
FILES_TO_TRANSFER = [
    # r"D:\zzr\server_fedavg\models\scalefl_modelutils.py"
    # r"D:\zzr\server_fedavg\client_heterofl_v1.py",      # 示例：更新单个脚本
    # r"D:\zzr\server_fedavg\utils",
    # r"D:\zzr\server_fedavg\client_SHFL.py",
    # r"D:\zzr\server_fedavg\test_shfl_distill_local.py",
    # r"D:\zzr\server_fedavg\models",                 # 示例：更新整个 models 文件夹
    # r"D:\zzr\科研\server_fedavg\utils\options.py",     # 你可以随时取消注释来添加
    # r"D:\zzr\server_fedavg\client_scalefl.py",
    r"D:\zzr\server_fedavg\utils\options.py",
]

# 2. 远程目标路径 (所有设备统一放这里)
REMOTE_PATH = "~/server_fedavg/utils/"  # 建议指定到具体项目文件夹，例如 ~/FedProject/

# 3. 设备列表 (IP, 用户名, 密码)
# 建议密码相同时直接复制粘贴，格式：(IP, 用户名, 密码)
DEVICES = [
    # --- Nano (7台) ---
    ("192.168.31.121", "jetsonnano", "admin"),
    ("192.168.31.244", "jetsonnano", "admin"),
    ("192.168.31.231", "jetsonnano", "admin"),
    ("192.168.31.237", "jetsonnano", "admin"),
    ("192.168.31.161", "jetsonnano", "admin"),
    ("192.168.31.239", "jetsonnano", "admin"),
    ("192.168.31.142", "jetsonnano", "admin"),

    ("192.168.31.243", "mastlab", "123"),
    ("192.168.31.198", "mastlab", "123"),

    ("192.168.31.19", "jetsonorin", "admin"),
    ("192.168.31.105", "yons", "123"),
]

# DEVICES = [
#     #更新server的时候用
#     ("192.168.31.105", "yons", "123"),

# ]

# ================= 核心逻辑 (无需修改) =================

def create_ssh_client(ip, user, pwd):
    """创建 SSH 连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username=user, password=pwd, timeout=5)
        return client
    except Exception as e:
        print(f"❌ 连接失败 {ip}: {e}")
        return None

def upload_files():
    print(f"🚀 开始批量部署...")
    print(f"📦 待传输文件: {len(FILES_TO_TRANSFER)} 个")
    print(f"🤖 目标设备: {len(DEVICES)} 台")
    print("-" * 50)

    for ip, user, pwd in DEVICES:
        print(f"正在连接 -> {user}@{ip} ... ", end="")
        ssh = create_ssh_client(ip, user, pwd)

        if ssh:
            try:
                # 建立 SCP 通道
                with SCPClient(ssh.get_transport()) as scp:
                    for file_path in FILES_TO_TRANSFER:
                        # 检查本地文件是否存在
                        if not os.path.exists(file_path):
                            print(f"\n   ⚠️ 本地文件不存在跳过: {file_path}")
                            continue

                        # 执行传输 (recursive=True 支持文件夹)
                        scp.put(file_path, remote_path=REMOTE_PATH, recursive=True)

                print("✅ 传输成功")
            except Exception as e:
                print(f"❌ 传输出错: {e}")
            finally:
                ssh.close()
        else:
            print("跳过")

    print("-" * 50)
    print("🎉 所有任务完成！")

if __name__ == "__main__":
    upload_files()