import time
import os
import sys
import numpy as np
import msvcrt
from datetime import datetime
import pickle
import argparse

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface

def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    hw = HardwareInterface(dummy_mode=args.dummy)
    if not hw.connect(): return

    try:
        radius = float(input("ボールの半径(mm): "))
        hw.ball_radius = radius
    except: hw.ball_radius = 0.0

    # Negative=Down 座標系でのホーム設定
    home_z = -15.0

    print(f"\nMoving to initial Home (Z={home_z}) and taring sensors...")
    hw.move_hand([90]*5)
    hw.cnc.move_to([0.0, 0.0, home_z])
    hw.wait_cnc()
    hw.tare_sensors()

    print("\n--- 記録モード (UFOキャッチャー) ---")
    print(f" [Space] : ホーム (Z={home_z}) / [Enter] : リフト & 保存")
    print(" [W] : 下へ (Minus) / [S] : 上へ (Plus)")
    print(" [A/D/Q/E] : 水平移動 / [ESC] : 保存して終了")

    # 現在の目標座標を初期化
    target_pos = [0.0, 0.0, home_z]
    
    recording = False
    current_traj = []

    try:
        while True:
            hw.update_sensor_data(update_cnc=False) # 通信をスキップして高速化
            master_angles = getattr(hw, '_latest_master_angles', [90]*5)
            
            # CNCの目標値を現在地として観測値に反映 (ラグ防止)
            hw._latest_cnc_pos[2] = target_pos[2]
            obs = hw.get_observation()
            
            # アクションの正規化 (角度5 + CNC Z 1)
            norm_angles = (np.array(master_angles) - 90.0) / 80.0
            cnc_min_z = -32.0; cnc_max_z = 0.0
            norm_z = (target_pos[2] - cnc_min_z) / (cnc_max_z - cnc_min_z) * 2.0 - 1.0
            action = np.concatenate([norm_angles, [norm_z]])

            # 記録中であればバッファに追加
            if recording:
                current_traj.append({"obs": obs, "acts": action})

            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b' ':
                    print(f"\nReset to Home (Z={home_z}). Starting recording..."); 
                    target_pos = [0.0, 0.0, home_z]
                    hw.move_hand([90]*5); hw.cnc.move_to(target_pos); hw.wait_cnc()
                    
                    # ホーム到達後にセンサーをゼロ点調整
                    hw.tare_sensors()
                    
                    # ホーム到達後に記録開始
                    recording = True
                    current_traj = []
                
                elif key == b'\r':
                    if not recording:
                        print("\n[Warning] Not recording. Press [Space] first.")
                        continue

                    recording = False

                    # 動きが止まるのを待ってからリフトアップ
                    hw.wait_cnc()
                    hw.update_sensor_data()

                    print("\n>>> Lifting 15mm for verification...")
                    success = hw.auto_lift(15.0)

                    if success:
                        print(f"Lift Success! Saving {len(current_traj)} steps.")
                        save_data([current_traj])
                    else:
                        print("Lift Failed. Data discarded.")

                    current_traj = []

                    # 自動的にホームに戻る (-15mm地点)
                    print(f"\nReturning to Home (Z={home_z})...")
                    target_pos = [0.0, 0.0, home_z]
                    hw.move_hand([90]*5)
                    hw.cnc.move_to(target_pos)
                    hw.wait_cnc()

                    print(" Press [Space] to start next recording.")

                elif key == b'w': 
                    target_pos[2] = max(-32.0, target_pos[2] - 1.0)
                elif key == b's': 
                    target_pos[2] = min(0.0, target_pos[2] + 1.0)
                elif key == b'a': target_pos[0] += 1.0
                elif key == b'd': target_pos[0] -= 1.0
                elif key == b'q': target_pos[1] += 1.0
                elif key == b'e': target_pos[1] -= 1.0
                elif key == b'\x1b': break
            
            # リアルタイム反映
            hw.move_hand(master_angles)
            hw.cnc.move_to(target_pos)
            hw.print_mms_status()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        hw.disconnect()
        print("\nCollection session ended.")

def save_data(data_log):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    filename = f"data/expert_demo_{timestamp}.pkl"
    with open(filename, 'wb') as f: pickle.dump(data_log, f)
    print(f"\nSaved {len(data_log)} samples to {filename}")

if __name__ == "__main__":
    collect()
