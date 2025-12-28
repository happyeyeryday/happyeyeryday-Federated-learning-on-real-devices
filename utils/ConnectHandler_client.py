import time
from loguru import logger
import pickle
import socket
import selectors
from tqdm import tqdm

class ConnectHandler(object):
    def __init__(self, HOST, POST, ID):
        self.socket = None
        self.addr = (HOST, POST)
        self.ID = ID
        self.register()

    def register(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info('connected to the server...')
        self.socket.connect(self.addr)
        logger.info("connected to the server successfully")
        logger.info("sending ID to server...")
        data = {"ID": self.ID}
        self.uploadToServer(data)
        logger.info("send completed")
        logger.debug("register completed")

    # def uploadToServer(self, data):
    #     # binary_data = pickle.dumps(data)
    #     # len_data = len(binary_data).to_bytes(8, byteorder="big")
    #     # length = len(binary_data)

    #     # binary_data = len_data + binary_data

    #     # logger.info("sending data ({} bytes) to client...", length)
    #     # # self.socket.sendall(binary_data)
    #     # logger.info("sending data ({} bytes) to client completely", length)
    #     try:
    #     # 设置发送超时
            
    #         binary_data = pickle.dumps(data)
    #         len_data = len(binary_data).to_bytes(8, byteorder="big")
    #         length = len(binary_data)

    #         binary_data = len_data + binary_data

    #         logger.info("开始发送数据 ({} bytes)...", length)
            
    #         # 分块发送大数据
    #         chunk_size = 1024 * 1024  # 1MB
    #         sent = 0
    #         for i in range(0, len(binary_data), chunk_size):
    #             chunk = binary_data[i:i+chunk_size]
    #             sent += self.socket.send(chunk)
    #             # 实时显示进度
    #             progress = (sent / len(binary_data)) * 100
    #             logger.info("已发送: {:.1f}% ({}/{} bytes)", progress, sent, len(binary_data))
                
    #         logger.info("数据发送完成 ({} bytes)", length)
            
    #         # 恢复默认无超时
    #         self.socket.settimeout(None)
            
    #     except socket.timeout:
    #         logger.error("发送数据超时！请检查服务器状态")
    #         raise
    #     except Exception as e:
    #         logger.error("发送数据时出错: {}", e)
    #         raise

    def uploadToServer(self, data):
        try:
            # === 探针 #A: 确认进入函数 ===
            logger.info("进入 uploadToServer 函数...")

            binary_data = pickle.dumps(data)
            length = len(binary_data)
            len_data = length.to_bytes(8, byteorder="big")
            
            # 先发送数据长度
            self.socket.sendall(len_data)

            # === 探针 #B: 确认长度已发送，将要发送主体数据 ===
            logger.info(f"数据长度({len(len_data)} bytes)已发送。准备发送主体数据({length} bytes)...")

            # 使用 sendall 来保证所有数据都被发送，这是最简单可靠的方式
            self.socket.sendall(binary_data)
            
            # === 探针 #C: 确认所有数据都已提交给操作系统缓冲区 ===
            # 注意：这仍然不代表服务器已接收完毕，但代表客户端的工作已完成。
            logger.success(f"所有数据 ({length} bytes) 已成功提交给网络缓冲区。uploadToServer 函数即将返回。")

        except Exception as e:
            logger.error("在 uploadToServer 中发生严重错误: {}", e)
            # 可以在这里加入重连或退出的逻辑
            raise
        
    def uploadToServerWithRateLimit(self, data, rate):
        binary_data = pickle.dumps(data)
        len_data = len(binary_data).to_bytes(8, byteorder="big")
        length = len(binary_data)

        binary_data = len_data + binary_data
        logger.info("sending data ({} bytes) to client...", length)
        total_sent = 0
        data_length = len(binary_data)
        while total_sent < data_length:
            conn.send(binary_data[total_sent:min(total_sent+rate, data_length)])
            total_sent = min(total_sent+rate, data_length)
        logger.info("sending data ({} bytes) to client completely", length)

    def receiveFromServer(self):
        total_length = int.from_bytes(self.socket.recv(8), byteorder="big")
        if total_length == 0:
            logger.critical("connection is closed by server!!! Server may crash!!!")
        logger.info("{} bytes data to be received".format(total_length))
        cur_length = 0
        total_data = bytes()
        pbar = tqdm(total=total_length, unit='iteration')
        # self.socket.settimeout(60)
        logger.info("start recving, timeout limit: 60")
        while cur_length < total_length:
            data = self.socket.recv(min(total_length-cur_length, 1024000))
            cur_length += len(data)
            total_data += data
            pbar.update(len(data))
        logger.info("receive completed")
        total_data = pickle.loads(total_data)
        # self.socket.settimeout(None)
        logger.info("end recving, timeout limit close")
        return total_data