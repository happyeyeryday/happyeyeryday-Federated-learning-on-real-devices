from pathlib import Path

import paramiko
from scp import SCPClient

from device_config import ALL_DEVICES

# ================= 配置区域 =================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "~/server_fedavg"

# 共享配置文件，同步给 server + 所有 client
FILES_TO_TRANSFER = [
    "utils/options.py",
    "utils/helcfl_real_profiles.py",
    "utils/power_manager_real.py",
    "utils/main_real_policy.py",
]

DEVICES = ALL_DEVICES


# ================= 核心逻辑 =================

def create_ssh_client(ip, user, pwd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username=user, password=pwd, timeout=5)
        return client
    except Exception as exc:
        print(f"❌ 连接失败 {ip}: {exc}")
        return None


def remote_dir_for(relative_path):
    relative_parent = Path(relative_path).parent.as_posix()
    if relative_parent == ".":
        return REMOTE_ROOT
    return f"{REMOTE_ROOT}/{relative_parent}"


def upload_files():
    print("🚀 开始同步共享配置...")
    print(f"📦 待传输文件: {len(FILES_TO_TRANSFER)} 个")
    print(f"🤖 目标设备: {len(DEVICES)} 台")
    print("-" * 50)

    for ip, user, pwd in DEVICES:
        print(f"正在连接 -> {user}@{ip} ... ", end="")
        ssh = create_ssh_client(ip, user, pwd)
        if ssh is None:
            print("跳过")
            continue

        try:
            with SCPClient(ssh.get_transport()) as scp:
                for relative_path in FILES_TO_TRANSFER:
                    local_path = PROJECT_ROOT / relative_path
                    if not local_path.exists():
                        print(f"\n   ⚠️ 本地文件不存在跳过: {local_path}")
                        continue

                    remote_dir = remote_dir_for(relative_path)
                    ssh.exec_command(f'mkdir -p "{remote_dir}"')
                    scp.put(
                        str(local_path),
                        remote_path=remote_dir,
                        recursive=local_path.is_dir(),
                    )
            print("✅ 传输成功")
        except Exception as exc:
            print(f"❌ 传输出错: {exc}")
        finally:
            ssh.close()

    print("-" * 50)
    print("🎉 所有任务完成！")


if __name__ == "__main__":
    upload_files()
