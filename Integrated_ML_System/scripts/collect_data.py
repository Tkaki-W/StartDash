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
        radius = float(input("ボールの半径(mm): "))
        hw.ball_radius = radius
    except: 
        radius = 0.0
        hw.ball_radius = 0.0

    hardness = input("ボールの硬さ (soft / hard): ").lower()
    if hardness not in ['soft', 'hard']:
        hardness = "unknown"

    # Negative=Down 座標系でのホーム設定
    home_z = -15.0

    print("\n--- 記録モード (UFOキャッチャー) ---")
    print(f" [Space] : ホーム (Z={home_z}) / [Enter] : リフト & 保存")
    print(" [W] : 下へ (Minus) / [S] : 上へ (Plus)")
    print(" [A/D/Q/E] : 水平移動 / [ESC] : 保存して終了")

    # 現在の目標座標を初期化
    target_pos = [0.0, 0.0, home_z]
    
    recording = False
    current_traj = []

    # 記録タイミング管理用
    last_record_time = 0
    record_interval = 0.05 # 20Hz

    try:
        while True:
            current_time = time.time()
            # 1. センサーデータと「現在の状態(Observation)」を取得
            # 手動で _latest_cnc_pos を上書きするのを止め、実際の状態を反映させる
            hw.update_sensor_data(update_cnc=False)
            obs = hw.get_observation()
            
            # アクションの正規化 (角度5 + CNC Z 1)
            norm_angles = (np.array(master_angles) - 90.0) / 80.0
            cnc_min_z = -35.0; cnc_max_z = 0.0
            norm_z = (target_pos[2] - cnc_min_z) / (cnc_max_z - cnc_min_z) * 2.0 - 1.0
            action = np.concatenate([norm_angles, [norm_z]])

            # 2. ユーザー入力を受け取り、「次の目標(target_pos)」を決定する
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b' ':
                    print(f"\nReset to Home. Starting recording..."); 
                    target_pos = [0.0, 0.0, home_z]
                    hw.move_hand([90]*5); hw.cnc.move_to(target_pos); hw.wait_cnc()
                    # ホーム到達後に記録開始
                    recording = True
                    current_traj = []
                    last_record_time = 0
                    continue # リセット時は記録をスキップ
                
                elif key == b'\r':
                    if not recording: continue
                    recording = False
                    hw.wait_cnc()
                    hw.update_sensor_data()
                    print("\n>>> Lifting 15mm for verification...")
                    hw.auto_lift(15.0)

                    # ユーザーに保存するか確認
                    print(f"\nSave these {len(current_traj)} steps? (y/n): ", end="", flush=True)
                    while True:
                        if msvcrt.kbhit():
                            choice = msvcrt.getch().lower()
                            if choice == b'y':
                                print("Yes! Saved."); 
                                save_data([current_traj]) # リストで包んで保存
                                break
                            elif choice == b'n':
                                print("No. Discarded."); break
                        time.sleep(0.01)

                    current_traj = []
                    
                                    # 自動的にホームに戻る (-15mm地点)
                    print(f"\nReturning to Home (Z={home_z})...")
                    target_pos = [0.0, 0.0, home_z]
                    hw.move_hand([90]*5)
                    hw.cnc.move_to(target_pos)
                    hw.wait_cnc()

                    print(" Press [Space] to start next recording.")

                elif key == b'w': 
                    target_pos[2] = max(-35.0, target_pos[2] - 1.0)
                elif key == b's': 
                    target_pos[2] = min(0.0, target_pos[2] + 1.0)
                elif key == b'a': target_pos[0] += 1.0
                elif key == b'd': target_pos[0] -= 1.0
                elif key == b'q': target_pos[1] += 1.0
                elif key == b'e': target_pos[1] -= 1.0
                elif key == b'\x1b': break

            # 3. 入力「後」の新しい目標値を「アクション」として記録する
            norm_angles = (np.array(master_angles) - 90.0) / 80.0
            cnc_min_z = -32.0; cnc_max_z = 0.0
            norm_z = (target_pos[2] - cnc_min_z) / (cnc_max_z - cnc_min_z) * 2.0 - 1.0
            action = np.concatenate([norm_angles, [norm_z]])

            # 記録中であればバッファに追加 (20Hz で記録)
            if recording and (current_time - last_record_time >= record_interval):
                should_record = False
                if len(current_traj) == 0:
                    should_record = True
                else:
                    last_action = current_traj[-1]["acts"]
                    # アクションに変化があるか、あるいは一定周期で強制記録
                    if np.max(np.abs(action - last_action)) > 0.005:
                        should_record = True
                    elif current_time - last_record_time > 0.2: # 動きがなくても0.2秒に1回は記録
                        should_record = True
                
                if should_record:
                    current_traj.append({"obs": obs, "acts": action})
                    last_record_time = current_time

            # 4. リアルタイム反映 (同期送信) - ここは常に高速に実行
            moved = hw.move_sync(master_angles, target_pos[2])
            
            # 5. 実際に動かした場合のみ到達を待機 (高速化のため、記録時以外は待たない)
            # if moved:
            #     hw.wait_reach_z(target_pos[2])
            
            hw.update_sensor_data(update_cnc=True)
            hw.print_mms_status()
            time.sleep(0.005) # ループ自体は高速(200Hz程度)にして操作感を滑らかにする

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        hw.disconnect()
        print("\nCollection session ended.")

def save_data(data_log, radius, hardness):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    filename = f"data/{int(radius)}mm_{hardness}_{timestamp}.pkl"
    with open(filename, 'wb') as f: pickle.dump(data_log, f)
    print(f"\nSaved {len(data_log)} samples to {filename}")

if __name__ == "__main__":
    collect()
