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

    if not args.dummy and hw.master_ser is None:
        print("\n[ERROR] データ収集にはマスター手袋が必要です。接続を確認してください。")
        hw.disconnect()
        return

    try:
        radius = float(input("物体のサイズ(mm): "))
        hw.ball_radius = radius
    except:
        radius = 0.0
        hw.ball_radius = 0.0

    shape = input("物体の形状 (ball / cube): ").lower()
    if shape not in ['ball', 'cube']:
        shape = "unknown"

    # Negative=Down 座標系でのホーム設定
    home_z = 0.0

    print(f"\nMoving to initial Home (Z={home_z}) and taring sensors...")
    hw.move_hand([90]*5)
    hw.cnc.move_to([0.0, 0.0, home_z])
    hw.wait_cnc()
    hw.update_sensor_data(update_cnc=True) # 現在地を正しく同期
    hw.tare_sensors()

    print("\n--- 記録モード (UFOキャッチャー) ---")
    print(f" [Space] : ホーム (Z={home_z}) / [Enter] : リフト & 保存")
    print(" [W] : 下へ (Minus) / [S] : 上へ (Plus)")
    print(" [A/D/Q/E] : 水平移動 / [ESC] : 保存して終了")

    # 現在の目標座標を初期化
    target_pos = [0.0, 0.0, home_z]
    
    recording = False
    current_traj = []

    # 状態管理
    # 0: 待機, 1: アプローチ中 (REACH), 2: 把持中 (GRASP)
    phase = 0
    reach_traj = []
    grasp_traj = []
    
    last_record_time = 0
    record_interval = 0.05

    try:
        while True:
            current_time = time.time()
            hw.update_sensor_data(update_cnc=False)
            obs = hw.get_observation()
            master_angles = getattr(hw, '_latest_master_angles', [90]*5)

            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b' ':
                    print(f"\n[PHASE: REACH] Starting Reach Phase. Move CNC near ball..."); 
                    target_pos = [0.0, 0.0, home_z]
                    hw.move_hand([90]*5); hw.cnc.move_to(target_pos); hw.wait_cnc()
                    hw.tare_sensors()
                    phase = 1
                    reach_traj = []; grasp_traj = []
                    last_record_time = 0
                    continue
                
                elif key == b'\r':
                    if phase == 1:
                        print(f"\n[PHASE: GRASP] Reach finished ({len(reach_traj)} steps). Now grasp the ball!")
                        phase = 2
                    elif phase == 2:
                        print(f"\n[DONE] Grasp finished ({len(grasp_traj)} steps). Verifying lift...")
                        hw.wait_cnc()
                        success = hw.auto_lift(15.0)
                        if success:
                            save_dual_data(reach_traj, grasp_traj, radius, shape)
                        else:
                            print("Lift Failed. Data discarded.")
                        phase = 0
                    continue

                elif key == b'w': target_pos[2] = max(-32.0, target_pos[2] - 1.0)
                elif key == b's': target_pos[2] = min(0.0, target_pos[2] + 1.0)
                elif key == b'a': target_pos[0] += 1.0
                elif key == b'd': target_pos[0] -= 1.0
                elif key == b'q': target_pos[1] += 1.0
                elif key == b'e': target_pos[1] -= 1.0
                elif key == b'\x1b': break

            # アクション生成 (AI入力ルール +1.0=閉, -1.0=開)
            norm_angles = np.zeros(5)
            norm_angles[0] = (master_angles[0] - 95.0) / 75.0
            norm_angles[1:] = -(np.array(master_angles[1:]) - 100.0) / 80.0
            
            # Zの正規化 (一発降下用)
            norm_z = (target_pos[2] / 16.0) + 1.0

            # 記録処理 (20Hz)
            if phase > 0 and (current_time - last_record_time >= record_interval):
                if phase == 1:
                    # REACHフェーズ: obsは1次元(半径のみ), actsは1次元(目標Zのみ)
                    reach_traj.append({
                        "obs": np.array([hw.ball_radius / 10.0]), # ミニマム保存
                        "acts": np.array([norm_z])
                    })
                    moved = hw.move_sync([90]*5, target_pos[2])
                else:
                    # GRASPフェーズ: obsは9次元, actsは5次元
                    grasp_traj.append({"obs": obs, "acts": norm_angles})
                    moved = hw.move_sync(master_angles, target_pos[2])
                last_record_time = current_time
            else:
                # 記録タイミング以外
                current_hand_target = [90]*5 if phase == 1 else master_angles
                moved = hw.move_sync(current_hand_target, target_pos[2])
            hw.update_sensor_data(update_cnc=False)
            hw.print_mms_status(prefix=f"[{'REACH' if phase==1 else 'GRASP' if phase==2 else 'IDLE'}]")
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        hw.disconnect()
        print("\nCollection session ended.")

def save_dual_data(reach_traj, grasp_traj, radius, shape):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)

    reach_file = f"data/{int(radius)}mm_{shape}_reach_{timestamp}.pkl"
    grasp_file = f"data/{int(radius)}mm_{shape}_grasp_{timestamp}.pkl"
    
    with open(reach_file, 'wb') as f: pickle.dump([reach_traj], f)
    with open(grasp_file, 'wb') as f: pickle.dump([grasp_traj], f)
    
    print(f"\nSaved Reach: {len(reach_traj)} steps, Grasp: {len(grasp_traj)} steps.")
    print(f"Files: {reach_file}, {grasp_file}")

if __name__ == "__main__":
    collect()
