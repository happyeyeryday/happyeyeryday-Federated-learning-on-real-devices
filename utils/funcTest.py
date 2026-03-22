import time, multiprocessing


def work1():
    for i in range(5):
        print("正在运行 work1...")
        time.sleep(0.5)


if __name__ == '__main__':
    process_obj = multiprocessing.Process(target=work1)  # 创建子进程对象
    process_obj.start()  # 启动进程
    for _ in range(5):
        print("主进程同时在运行...")
        time.sleep(1.0)
