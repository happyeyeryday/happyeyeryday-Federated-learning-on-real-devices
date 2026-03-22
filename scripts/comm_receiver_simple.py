#!/usr/bin/env python3
"""
极简接收端脚本。

默认只负责：
1. 接收长度头和 payload
2. 收满 payload
3. 回复 b"ok"

运行：
python3 scripts/comm_receiver_simple.py
"""

import csv
import os
import socket
from datetime import datetime


HOST = "0.0.0.0"
PORT = 8080
BUFFER_SIZE = 1024 * 1024
OUTPUT_DIR = os.path.join("logs", "comm_energy")
LOG_FILE = os.path.join(OUTPUT_DIR, "receiver_latest.csv")


def now_text():
    return datetime.now().isoformat(timespec="microseconds")


def ensure_log_file():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(LOG_FILE):
        return
    with open(LOG_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "peer", "bytes", "status", "error"])


def append_log(row):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(row)


def recv_exact(conn, total_bytes):
    data = bytearray()
    while len(data) < total_bytes:
        chunk = conn.recv(min(BUFFER_SIZE, total_bytes - len(data)))
        if not chunk:
            raise ConnectionError(
                f"connection closed while receiving {len(data)}/{total_bytes} bytes"
            )
        data.extend(chunk)
    return bytes(data)


def main():
    ensure_log_file()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(8)
        print(f"receiver listening on {HOST}:{PORT}", flush=True)

        while True:
            conn, addr = server_sock.accept()
            with conn:
                status = "ok"
                error = ""
                payload_bytes = 0

                try:
                    total_length = int.from_bytes(recv_exact(conn, 8), byteorder="big")
                    payload = recv_exact(conn, total_length)
                    payload_bytes = len(payload)
                    conn.sendall(b"ok")
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    error = str(exc)

                peer = f"{addr[0]}:{addr[1]}"
                timestamp = now_text()
                append_log([timestamp, peer, payload_bytes, status, error])
                print(
                    f"time={timestamp} peer={peer} bytes={payload_bytes} status={status}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
