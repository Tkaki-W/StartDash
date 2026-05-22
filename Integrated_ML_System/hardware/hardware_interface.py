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

class HardwareInterface:
    def __init__(self, master_port="COM5", slave_port="COM4", cnc_port="COM10", mms_port="COM9", dummy_mode=False):
        self.master_port = master_port
        self.slave_port = slave_port
        self.cnc_port = cnc_port
        self.mms_port = mms_port
        self.dummy_mode = dummy_mode

        self.master_ser = None
        self.slave_ser = None
        self.cnc = None
        self.mms_sensor = None

        self.ball_radius = 0.0
        self.lock = threading.Lock()
        
        # 内部状態
        self._latest_mms_data = np.zeros(6)
        self._latest_slave_angles = [90.0] * 5 
        self._latest_cnc_pos = [0.0, 0.0, -15.0] 
        self._last_sent_angles = [0.0] * 5
        
        # 接触判定用
        self.mms_history = deque()
        self.contact_status = 0 # 0: 非接触, 1: 接触

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
                self.cnc.transaction_wait = 0.01
            self.mms_sensor = MMS101ForceSensor(self.mms_port, baudrate=1000000, output6=True)
            if self.mms_sensor.is_opened():
                self.mms_sensor.start_continuous_read()
                time.sleep(1); self.mms_sensor.set_zero()
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
                                self.arduino_map(data[0], 300, 800, 10, 170),
                                180 - self.arduino_map(data[1], 400, 600, 0, 160),
                                180 - self.arduino_map(data[2], 400, 600, 0, 160),
                                180 - self.arduino_map(data[3], 400, 600, 0, 160),
                                180 - self.arduino_map(data[4], 400, 600, 0, 160)
                            ]
                            break
                except: pass

            if self.mms_sensor and self.mms_sensor.is_opened():
                mms_list = self.mms_sensor.get_data()
                if mms_list:
                    now = time.time()
                    for new_data, ts in mms_list:
                        # 6軸データの平均を取る (read_mms101.pyのロジックに合わせる)
                        avg_force = np.mean(new_data, axis=0)
                        self._latest_mms_data = avg_force
                        current_fz = avg_force[2]

                        # 履歴に追加
                        self.mms_history.append((now, current_fz))

                        # 0.2秒より古いデータを削除 (判定時間を短縮)
                        while self.mms_history and self.mms_history[0][0] < now - 0.2:
                            past_time, past_fz = self.mms_history.popleft()
                            diff = current_fz - past_fz

                            # 判定ロジック
                            if diff <= -0.1:
                                self.contact_status = 1 # 接触
                            elif diff > 0.1:
                                self.contact_status = 0 # 非接触

            # CNC位置 (通信が重いため必要な時だけ更新)
            if self.cnc and update_cnc:
                pos, _ = self.cnc.get_current_pos()
                self._latest_cnc_pos = pos

    def get_observation(self):
        with self.lock:
            norm_angles = (np.array(self._latest_slave_angles) - 90.0) / 80.0
            # 3(MMS Fx,Fy,Fz) + 1(CNC Z) + 1(Radius) + 5(Angles) = 10次元
            return np.concatenate([
                self._latest_mms_data[:3], 
                [self._latest_cnc_pos[2]], 
                [self.ball_radius], 
                norm_angles
            ]).astype(np.float32)

    def set_ball_radius(self, radius):
        self.ball_radius = float(radius)

    def print_mms_status(self):
        with self.lock:
            status_str = "接触" if self.contact_status == 1 else "非接触"
            mms = self._latest_mms_data
            sys.stdout.write(f"\r [{status_str}] Fx:{mms[0]:6.2f} Fy:{mms[1]:6.2f} Fz:{mms[2]:6.2f} Mx:{mms[3]:6.2f} My:{mms[4]:6.2f} Mz:{mms[5]:6.2f}")
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
        """15mmリフトアップ (マイナス世界での上昇)"""
        # 現在地 (例: -30.0) に 距離(15.0) を足して原点に近づける (-15.0)
        current_z = self._latest_cnc_pos[2]
        target_z = min(0.0, current_z + abs(dist)) 
        
        print(f"--- Auto-Lift: {current_z:.1f} -> {target_z:.1f} ---")
        self.cnc.move_to([self._latest_cnc_pos[0], self._latest_cnc_pos[1], target_z])
        self.wait_cnc()
        self.update_sensor_data()
        return self._latest_mms_data[2] > 1.0

    def arduino_map(self, x, in_min, in_max, out_min, out_max):
        val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return int(max(min(out_min, out_max), min(max(out_min, out_max), val)))
