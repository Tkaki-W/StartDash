import serial as pyserial
import time
import csv
import msvcrt
import os
from datetime import datetime

class SerialCommunicator:
    """
    マスター(手袋)からデータを受け取り、
    スレーブ(ロボット手)へ送信すると同時に
    スレーブからの触覚フィードバックを受け取って保存するクラス
    """
    def __init__(self, rx_port, rx_baud_rate, tx_port, tx_baud_rate):
        self.rx_port = rx_port # Master (COM3)
        self.rx_baud_rate = rx_baud_rate
        self.tx_port = tx_port # Slave (COM4)
        self.tx_baud_rate = tx_baud_rate
        self.rx = None
        self.tx = None
        self.data_history = []
        self.is_recording = False

    def connect(self):
        try:
            # タイムアウトを適切に設定し、レスポンスを確保
            self.rx = pyserial.Serial(self.rx_port, self.rx_baud_rate, timeout=1)
            self.tx = pyserial.Serial(self.tx_port, self.tx_baud_rate, timeout=1)
            print(f"Connected to Master={self.rx_port}, Slave={self.tx_port}")
            return True
        except Exception as e:
            print(f"接続失敗: {e}")
            return False

    def disconnect(self):
        if self.rx: self.rx.close()
        if self.tx: self.tx.close()
        print("Serial port closed.")

    def arduino_map(self, x, master_min, master_max, slave_min, slave_max):
        """Arduinoのmap関数相当"""
        val = (x - master_min) * (slave_max - slave_min) / (master_max - master_min) + slave_min
        if slave_min < slave_max:
            return int(max(slave_min, min(slave_max, val)))
        else:
            return int(max(slave_max, min(slave_min, val)))

    def communicate(self):
        if not self.rx or not self.tx:
            print("未接続です")
            return

        print("-" * 30)
        print(" [Space] キー: 記録の開始/停止")
        print(" [Ctrl+C]    : 終了して保存")
        print("-" * 30)

        # 最後に受信した触覚センサー値
        current_tactile = 0.0

        try:
            while True:
                # --- 1. キー入力チェック ---
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':
                        self.is_recording = not self.is_recording
                        status = "● 記録中..." if self.is_recording else "■ 停止中 (待機)"
                        print(f"\n{status}")

                # --- 2. マスターからデータを受信 (s0~s4) ---
                line_m = self.rx.readline().decode('utf-8', errors='ignore').strip()

                if line_m:
                    try:
                        finger_data = [float(x) for x in line_m.split(',')]

                        if len(finger_data) >= 5:
                            # 角度に変換
                            angles = [
                                self.arduino_map(finger_data[0], 300, 800, 10, 170),
                                180 - self.arduino_map(finger_data[1], 300, 600, 10, 160),
                                180 - self.arduino_map(finger_data[2], 300, 600, 10, 160),
                                180 - self.arduino_map(finger_data[3], 200, 500, 10, 160),
                                180 - self.arduino_map(finger_data[4], 400, 600, 10, 160)
                            ]

                            # --- 3. スレーブへ角度を送信 ---
                            send_data = ",".join(map(str, angles)) + "\n"
                            self.tx.write(send_data.encode())
                            
                            # --- 4. スレーブから触覚値を読み取り (t0) ---
                            # 直後の返信を待機
                            line_s = self.tx.readline().decode('utf-8', errors='ignore').strip()
                            if line_s:
                                try:
                                    current_tactile = float(line_s)
                                    print(current_tactile)
                                except ValueError:
                                    pass

                            # --- 5. データの記録 ---
                            if self.is_recording:
                                # [s0, s1, s2, s3, s4, t0, a0, a1, a2, a3, a4] の形式
                                record = finger_data[:5] + [current_tactile] + angles
                                self.data_history.append(record)
                                print(".", end="", flush=True)
                            
                        else:
                            print(f"\n受信データ不足: {finger_data}")

                    except ValueError:
                        print(f"\nスキップ(変換エラー): {line_m}")

        except KeyboardInterrupt:
            print("\n終了リクエストを受け付けました。")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.disconnect()

    def save_to_csv(self):
        if not self.data_history:
            print("保存するデータがありません。")
            return

        # 'data' フォルダのパス
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(data_dir, f"tactile_training_data_{timestamp}.csv")

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # s0~s4: マスター, t0: 触覚(スレーブ), a0~a4: サーボ角
                header = [f"s{i}" for i in range(5)] + ["t0"] + [f"a{i}" for i in range(5)]
                writer.writerow(header)
                writer.writerows(self.data_history)
            
            print(f"\nデータを {filename} に保存しました。")
            print(f"合計データ数: {len(self.data_history)} サンプル")
        except Exception as e:
            print(f"CSV保存失敗: {e}")

if __name__ == "__main__":
    # COMポートは環境に合わせて設定してください
    # (例: Master=COM3, Slave=COM4)
    master = SerialCommunicator("COM5", 9600, "COM4", 9600)
    if master.connect():
        master.communicate()
        master.save_to_csv()
