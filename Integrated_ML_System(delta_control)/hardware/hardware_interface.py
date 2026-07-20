import serial
import time
import sys
import os
import threading
import numpy as np
from collections import deque

# 親ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from FSCal.stage_ctrl import GRBLControl
    from FSCal.FS_MMS101 import MMS101ForceSensor
except ImportError:
    class GRBLControl:
        def __init__(self, *args, **kwargs): self.cur_pos = [0, 0, -15.0]; self.in_motion = False
        def is_port_opened(self): return False
        def transaction(self, *args): return "ok"
        def set_origin(self): self.cur_pos = [0, 0, 0]
        def get_current_pos(self): return self.cur_pos, [0, 0, 0]
        def move_to(self, pos): self.cur_pos = list(pos)
    class MMS101ForceSensor:
        def __init__(self, *args, **kwargs): pass
        def is_opened(self): return False
        def start_continuous_read(self): pass
        def stop_continuous_read(self): pass
        def set_zero(self): pass
        def get_data(self): return []


#mms1,2,3がそれぞれ親指、薬指、人差し指に対応しちゃってるので気を付けてね( ；∀；)(学習モデルを一からつくる際は気にしなくていいと思うけど)
class HardwareInterface:
    def __init__(self, master_port="COM5", slave_port="COM4", cnc_port="COM6", mms_port="COM9", mms2_port="COM14", mms3_port="COM15", dummy_mode=False):
        self.master_port = master_port
        self.slave_port = slave_port
        self.cnc_port = cnc_port
        self.mms_port = mms_port
        self.mms2_port = mms2_port
        self.mms3_port = mms3_port
        self.dummy_mode = dummy_mode

        self.master_ser = None
        self.slave_ser = None
        self.cnc = None
        self.mms_sensor = None
        self.mms_sensor2 = None
        self.mms_sensor3 = None

        self.ball_radius = 0.0
        self.lock = threading.Lock()
        
        # 内部状態
        self._latest_mms_data = np.zeros(6)
        self._latest_mms2_data = np.zeros(6)
        self._latest_mms3_data = np.zeros(6)
        self._latest_slave_angles = [90.0] * 5 
        self._latest_cnc_pos = [0.0, 0.0, -15.0] 
        self._last_sent_angles = [0.0] * 5
        
        # 接触判定用
        self.last_fz_values = [0.0] * 3
        self.contact_statuses = [0] * 3 

    def connect(self):
        if self.dummy_mode:
            print("[DUMMY] Connected."); return True
        try:
            # マスター手袋 (オプション)
            try:
                self.master_ser = serial.Serial(self.master_port, 9600, timeout=0.001)
                print(f"Master hand connected on {self.master_port}")
            except Exception as e:
                print(f"Warning: Could not connect to Master hand on {self.master_port}. (Optional for non-collection modes)")
                self.master_ser = None

            # スレーブハンド (必須)
            self.slave_ser = serial.Serial(self.slave_port, 9600, timeout=0.001)
            
            # CNC (必須)
            self.cnc = GRBLControl(self.cnc_port, 115200)
            if self.cnc.is_port_opened():
                self.cnc.transaction("$X")
                self.cnc.set_origin()
                self.cnc.transaction_wait = 0.001
            
            # センサー1 (COM9)
            self.mms_sensor = MMS101ForceSensor(self.mms_port, baudrate=1000000, output6=True)
            if self.mms_sensor.is_opened():
                self.mms_sensor.start_continuous_read()
                time.sleep(0.5); self.mms_sensor.set_zero()
            
            # センサー2 (COM14)
            self.mms_sensor2 = MMS101ForceSensor(self.mms2_port, baudrate=1000000, output6=True)
            if self.mms_sensor2.is_opened():
                self.mms_sensor2.start_continuous_read()
                time.sleep(0.5); self.mms_sensor2.set_zero()

            # センサー3 (COM15)
            self.mms_sensor3 = MMS101ForceSensor(self.mms3_port, baudrate=1000000, output6=True)
            if self.mms_sensor3.is_opened():
                self.mms_sensor3.start_continuous_read()
                time.sleep(0.5); self.mms_sensor3.set_zero()

            return True
        except Exception as e:
            print(f"Error in Hardware Connection: {e}"); return False

    def disconnect(self):
        if self.dummy_mode: return
        if self.cnc:
            try:
                # 停止指令
                self.cnc.transaction("!") 
                time.sleep(0.01)
                self.cnc.transaction("\x18")
            except: pass

        if self.master_ser: self.master_ser.close()
        if self.slave_ser: self.slave_ser.close()
        if self.mms_sensor and self.mms_sensor.is_opened():
            self.mms_sensor.stop_continuous_read()
        if self.mms_sensor2 and self.mms_sensor2.is_opened():
            self.mms_sensor2.stop_continuous_read()
        if self.mms_sensor3 and self.mms_sensor3.is_opened():
            self.mms_sensor3.stop_continuous_read()

    def _process_mms_data(self, sensor_idx, mms_list):
        if not mms_list: return
        # 溜まっているデータを一気に処理せず、最新のもののみ抽出して処理効率化
        new_data, ts = mms_list[-1]
        avg_force = np.mean(new_data, axis=0)
        if sensor_idx == 0: self._latest_mms_data = avg_force
        elif sensor_idx == 1: self._latest_mms2_data = avg_force
        elif sensor_idx == 2: self._latest_mms3_data = avg_force
        
        current_fz = avg_force[2]
        
        # 直前の値との差分で判定
        diff = current_fz - self.last_fz_values[sensor_idx]
        self.last_fz_values[sensor_idx] = current_fz

        # 状態保持型ロジック：
        if self.contact_statuses[sensor_idx] == 0:
            if diff <= -0.1: 
                self.contact_statuses[sensor_idx] = 1 # 接触
        else:
            if diff >= 0.05: 
                self.contact_statuses[sensor_idx] = 0 # 非接触

    def update_sensor_data(self, update_cnc=False):
        with self.lock:
            if self.dummy_mode: return
            # マスター手袋の読み取り (溜まっているバッファをパージして最新のみ取得)
            if self.master_ser and self.master_ser.in_waiting > 0:
                try:
                    raw = self.master_ser.read(self.master_ser.in_waiting).decode('utf-8', errors='ignore')
                    lines = raw.strip().split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line.count(',') >= 4:
                            data = [float(x) for x in line.split(',')]
                            self._latest_master_angles = [
                                self.arduino_map(data[0], 400, 700, 20, 170),
                                180 - self.arduino_map(data[1], 300, 600, 0, 160),
                                180 - self.arduino_map(data[2], 300, 600, 0, 160),
                                180 - self.arduino_map(data[3], 200, 500, 0, 160),
                                180 - self.arduino_map(data[4], 300, 600, 0, 160)
                            ]
                            break
                except: pass

            # 各センサーのデータ処理
            if self.mms_sensor and self.mms_sensor.is_opened():
                self._process_mms_data(0, self.mms_sensor.get_data())
            if self.mms_sensor2 and self.mms_sensor2.is_opened():
                self._process_mms_data(1, self.mms_sensor2.get_data())
            if self.mms_sensor3 and self.mms_sensor3.is_opened():
                self._process_mms_data(2, self.mms_sensor3.get_data())

            if self.cnc and update_cnc:
                pos, _ = self.cnc.get_current_pos()
                self._latest_cnc_pos = pos

    def get_observation(self):
        with self.lock:
            # 各指の正規化を統一 (+1.0=閉, -1.0=開)
            raw_angles = np.array(self._latest_slave_angles)
            norm_angles = np.zeros(5, dtype=np.float32)
            norm_angles[0] = (raw_angles[0] - 95.0) / 75.0
            norm_angles[1:] = -(raw_angles[1:] - 100.0) / 80.0

            # MMS Fz の取得とドリフトガード (プラス方向は0固定、かつ -0.1N 未満の微細なノイズもカット)
            fz1 = self._latest_mms_data[2] if self._latest_mms_data[2] < -0.1 else 0.0
            fz2 = self._latest_mms2_data[2] if self._latest_mms2_data[2] < -0.1 else 0.0
            fz3 = self._latest_mms3_data[2] if self._latest_mms3_data[2] < -0.1 else 0.0

            return np.concatenate([
                [fz1], [fz2], [fz3],
                [self.ball_radius / 10.0], # 半径 (3mmなら0.3)
                norm_angles
            ]).astype(np.float32)

    def tare_sensors(self):
        if self.dummy_mode: return
        with self.lock:
            if self.mms_sensor and self.mms_sensor.is_opened():
                self.mms_sensor.set_zero()
            if self.mms_sensor2 and self.mms_sensor2.is_opened():
                self.mms_sensor2.set_zero()
            if self.mms_sensor3 and self.mms_sensor3.is_opened():
                self.mms_sensor3.set_zero()
            
            self.last_fz_values = [0.0] * 3
            self.contact_statuses = [0] * 3
            print("\n--- 触覚センサー初期化完了 ---", flush=True)
            time.sleep(0.5)

    def print_mms_status(self, prefix=""):
        with self.lock:
            s_flags = ["触" if s == 1 else "空" for s in self.contact_statuses]
            m1 = self._latest_mms_data
            m2 = self._latest_mms2_data
            m3 = self._latest_mms3_data
            angles = [int(a) for a in self._latest_slave_angles]
            sys.stdout.write(f"\r{prefix} [S1:{s_flags[0]} S2:{s_flags[1]} S3:{s_flags[2]}] Fz1:{m1[2]:5.2f} Fz2:{m2[2]:5.2f} Fz3:{m3[2]:5.2f} Z:{self._latest_cnc_pos[2]:5.1f} Deg:{angles}")
            sys.stdout.flush()

    def move_hand(self, angles):
        if angles is None: return
        self._latest_slave_angles = list(angles)
        if self.dummy_mode or not self.slave_ser: return
        self.slave_ser.write((",".join(map(str, angles)) + "\n").encode())
        self._last_sent_angles = list(angles)

    def move_sync(self, angles, target_z, force_send=False):
        if angles is None: return False
        
        # 安全ガード: Z軸は 0.0 を超えない
        target_z = min(0.0, float(target_z))
        
        angle_diff = np.max(np.abs(np.array(angles) - np.array(self._last_sent_angles)))
        z_diff = abs(target_z - self._latest_cnc_pos[2])

        cnc_sent = False
        if force_send or angle_diff > 1.0 or z_diff > 0.1:
            self.move_hand(angles)
            if self.cnc and (force_send or z_diff > 0.05):
                self.cnc.jog([self._latest_cnc_pos[0], self._latest_cnc_pos[1], target_z])
                self._latest_cnc_pos[2] = target_z
                cnc_sent = True
        return cnc_sent

    def wait_reach_z(self, target_z, tolerance=0.1, timeout=0.5):
        """CNCの高さが目標値に到達するまで待機する"""
        if self.dummy_mode or not self.cnc: return
        start_t = time.time()
        while time.time() - start_t < timeout:
            pos, _ = self.cnc.get_current_pos()
            if abs(pos[2] - target_z) <= tolerance:
                break
            time.sleep(0.01)

    def wait_cnc(self):
        if self.dummy_mode or not self.cnc: return
        time.sleep(0.2)
        while True:
            self.cnc.get_current_pos()
            if not self.cnc.in_motion: break
            time.sleep(0.1)

    def auto_lift(self, dist=15.0):
        current_z = self._latest_cnc_pos[2]
        target_z = min(0.0, current_z + abs(dist)) 
        print(f"\n--- Auto-Lift: {current_z:.1f} -> {target_z:.1f} ---")
        self.cnc.move_to([self._latest_cnc_pos[0], self._latest_cnc_pos[1], target_z])
        self.wait_cnc()
        self.update_sensor_data()
        while True:
            ans = input("持ち上げに成功しましたか？ (y:成功 / n:失敗): ").lower()
            if ans == 'y': return True
            elif ans == 'n': return False
            else: print("y か n で入力してください。")

    def arduino_map(self, x, in_min, in_max, out_min, out_max):
        val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return int(max(min(out_min, out_max), min(max(out_min, out_max), val)))
