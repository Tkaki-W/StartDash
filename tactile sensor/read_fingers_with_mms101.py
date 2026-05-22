import serial as pyserial
import time
import csv
import msvcrt
import os
import sys
import numpy as np
from datetime import datetime

# FSCalのパスを追加
fscal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FSCal'))
sys.path.append(fscal_path)

try:
    from FS_MMS101 import MMS101ForceSensor
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

class FingerMMS101Communicator:
    """
    マスター(手袋)からデータを受け取り、スレーブ(ロボット手)へ送信し、
    同時にMMS101から力覚データを受け取って表示・保存するクラス
    """
    def __init__(self, master_port, slave_port, mms_port):
        self.master_port = master_port
        self.slave_port = slave_port
        self.mms_port = mms_port
        
        self.master_ser = None
        self.slave_ser = None
        self.mms_sensor = None
        
        self.data_history = []
        self.is_recording = False

    def connect(self):
        try:
            # マスターとスレーブのシリアル接続
            self.master_ser = pyserial.Serial(self.master_port, 9600, timeout=1)
            self.slave_ser = pyserial.Serial(self.slave_port, 9600, timeout=1)
            print(f"Connected: Master={self.master_port}, Slave={self.slave_port}")
            
            # MMS101の初期化
            print(f"Initializing MMS101 on {self.mms_port}...")
            self.mms_sensor = MMS101ForceSensor(self.mms_port, baudrate=1000000, output6=True)
            if not self.mms_sensor.is_opened():
                print("MMS101のオープンに失敗しました。")
                return False
            
            self.mms_sensor.start_continuous_read()
            time.sleep(2) # 安定待ち
            self.mms_sensor.set_zero()
            print("MMS101 Ready.")
            
            return True
        except Exception as e:
            print(f"接続失敗: {e}")
            return False

    def disconnect(self):
        if self.master_ser: self.master_ser.close()
        if self.slave_ser: self.slave_ser.close()
        if self.mms_sensor:
            self.mms_sensor.stop_continuous_read()
        print("All connections closed.")

    def arduino_map(self, x, master_min, master_max, slave_min, slave_max):
        val = (x - master_min) * (slave_max - slave_min) / (master_max - master_min) + slave_min
        if slave_min < slave_max:
            return int(max(slave_min, min(slave_max, val)))
        else:
            return int(max(slave_max, min(slave_min, val)))

    def communicate(self):
        if not self.master_ser or not self.slave_ser or not self.mms_sensor:
            print("未接続です")
            return

        print("-" * 60)
        print(" [Space] キー: 記録の開始/停止")
        print(" [Ctrl+C]    : 終了して保存")
        print("-" * 60)
        print(" [Count]   Fx     Fy     Fz     Tx     Ty     Tz")

        total_samples = 0
        latest_mms_data = np.zeros(6)

        try:
            while True:
                # --- 1. キー入力チェック ---
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':
                        self.is_recording = not self.is_recording
                        status = "\n● 記録中..." if self.is_recording else "\n■ 停止中 (待機)"
                        print(status)

                # --- 2. MMS101から最新データ取得 ---
                mms_list = self.mms_sensor.get_data()
                if mms_list:
                    # 最新のチャンクの最新の1点を使用
                    latest_chunk, _ = mms_list[-1]
                    latest_mms_data = latest_chunk[-1]
                    
                    for chunk, _ in mms_list:
                        total_samples += len(chunk)

                    # 画面表示の更新
                    val_str = " ".join([f"{x:6.2f}" for x in latest_mms_data])
                    sys.stdout.write(f"\r[{total_samples:8d}] {val_str}")
                    sys.stdout.flush()

                # --- 3. マスターからデータを受信 ---
                line_m = self.master_ser.readline().decode('utf-8', errors='ignore').strip()
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

                            # --- 4. スレーブへ角度を送信 ---
                            send_data = ",".join(map(str, angles)) + "\n"
                            self.slave_ser.write(send_data.encode())
                            
                            # スレーブからの触覚値(t0)は今回無視して読み飛ばす
                            if self.slave_ser.in_waiting:
                                self.slave_ser.readline()

                            # --- 5. データの記録 ---
                            if self.is_recording:
                                # [s0~s4, MMS(6軸), a0~a4] の形式で保存
                                record = finger_data[:5] + list(latest_mms_data) + angles
                                self.data_history.append(record)
                    except ValueError:
                        pass

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n終了リクエストを受け付けました。")
        finally:
            self.disconnect()

    def save_to_csv(self):
        if not self.data_history:
            print("保存するデータがありません。")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(data_dir, f"mms101_finger_data_{timestamp}.csv")

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # ヘッダー: マスター(s), MMS101(m), サーボ角(a)
                header = ([f"s{i}" for i in range(5)] + 
                          ["m_fx", "m_fy", "m_fz", "m_tx", "m_ty", "m_tz"] + 
                          [f"a{i}" for i in range(5)])
                writer.writerow(header)
                writer.writerows(self.data_history)
            
            print(f"\nデータを {filename} に保存しました。")
            print(f"合計データ数: {len(self.data_history)} サンプル")
        except Exception as e:
            print(f"CSV保存失敗: {e}")

if __name__ == "__main__":
    # ポート設定 (Master=COM5, Slave=COM4, MMS101=COM9)
    # 環境に合わせて変更してください
    MASTER_PORT = "COM5"
    SLAVE_PORT = "COM4"
    MMS_PORT = "COM9"
    
    app = FingerMMS101Communicator(MASTER_PORT, SLAVE_PORT, MMS_PORT)
    if app.connect():
        app.communicate()
        app.save_to_csv()
