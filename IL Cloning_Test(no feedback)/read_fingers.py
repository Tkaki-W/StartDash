import serial as pyserial
import time
import csv
import msvcrt
import os
from datetime import datetime

class SerialCommunicator:

    def __init__(self, rx_port, rx_baud_rate, tx_port, tx_baud_rate):
        self.rx_port = rx_port
        self.rx_baud_rate = rx_baud_rate
        self.tx_port = tx_port
        self.tx_baud_rate = tx_baud_rate
        self.rx = None
        self.tx = None
        self.finger_data = None
        self.data_history = []
        # 記録状態のフラグ
        self.is_recording = False

    def connect(self):
        try:
            self.rx = pyserial.Serial(self.rx_port, self.rx_baud_rate, timeout=0.1)
            self.tx = pyserial.Serial(self.tx_port, self.tx_baud_rate, timeout=0.1)
            print(f"Connected to {self.rx_port} as rx, Connected to {self.tx_port} as tx")
        except Exception as e:
            print(f"接続失敗: {e}")

    def disconnect(self):
        if self.rx and self.rx.is_open:
            self.rx.close()
        if self.tx and self.tx.is_open:
            self.tx.close()
        print("Serial port closed.")

    def arduino_map(self, x, master_min, master_max, slave_min, slave_max):
        val = (x - master_min) * (slave_max - slave_min) / (master_max - master_min) + slave_min
        if slave_min < slave_max:
            return max(slave_min, min(slave_max, val))
        else:
            return max(slave_max, min(slave_min, val))

    def communicate(self):
        if self.rx is None or self.tx is None:
            print("未接続です")
            return

        print("-" * 30)
        print(" [Space] キー: 記録の開始/停止")
        print(" [Ctrl+C]    : 終了して保存")
        print("-" * 30)

        try:
            while True:
                # --- キー入力のチェック ---
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':  # スペースキー
                        self.is_recording = not self.is_recording
                        status = "● 記録中..." if self.is_recording else "■ 停止中 (待機)"
                        print(f"\n{status}")

                # --- シリアル通信処理 ---
                line = self.rx.readline().decode('utf-8').strip()

                if line:
                    try:
                        self.finger_data = [float(x) for x in line.split(',')]

                        if len(self.finger_data) >= 5:
                            angles = [
                                self.arduino_map(self.finger_data[0], 300, 600, 10, 180),
                                180 - self.arduino_map(self.finger_data[1], 400, 600, 0, 160),
                                180 - self.arduino_map(self.finger_data[2], 400, 600, 0, 160),
                                180 - self.arduino_map(self.finger_data[3], 400, 600, 0, 160),
                                180 - self.arduino_map(self.finger_data[4], 400, 600, 0, 160)
                            ]

                            send_data = ",".join(map(str, angles)) + "\n"
                            self.tx.write(send_data.encode())
                            
                            # 記録フラグがTrueの時だけ保存
                            if self.is_recording:
                                record = self.finger_data[:5] + angles
                                self.data_history.append(record)
                                # 記録中はドットを表示して動いていることを確認しやすくする
                                print(".", end="", flush=True)
                            
                        else:
                            print(f"\n受信データ不足: {self.finger_data}")

                    except ValueError:
                        print(f"\nスキップ(変換エラー): {line}")

        except KeyboardInterrupt:
            print("\n終了リクエストを受け付けました。")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.disconnect()

    def save_to_csv(self, filename=None):
        if not self.data_history:
            print("保存するデータがありません。記録せずに終了します。")
            return

        # スクリプトがあるディレクトリの直下に 'data' フォルダを指定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(data_dir, f"training_data_{timestamp}.csv")

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [f"s{i}" for i in range(5)] + [f"a{i}" for i in range(5)]
                writer.writerow(header)
                writer.writerows(self.data_history)
            
            print(f"\nデータを {filename} に保存しました。")
            print(f"合計データ数: {len(self.data_history)} サンプル")
        except Exception as e:
            print(f"CSV保存失敗: {e}")

if __name__ == "__main__":
    # COMポートはご自身の環境に合わせてください
    master = SerialCommunicator("COM5", 9600, "COM4", 9600)
    master.connect()
    master.communicate()
    #master.save_to_csv()
