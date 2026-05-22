import pickle
import os
import sys
import numpy as np

def dump_trajectory(filename):
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    file_path = os.path.join(base_dir, "data", filename)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to read: {e}")
        return
    
    traj = data[0] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) else data

    print(f"\n--- Detailed Dump: {filename} ---")
    print(f"Steps: {len(traj)}")
    print("-" * 115)
    # ヘッダー (Obsの主要項目とAct)
    print(f"{'Step':<4} | {'Z_mm':<6} | {'Fz1':<5} {'Fz2':<5} {'Fz3':<5} | {'Norm_Z':<6} | {'AI_Angles (T...Other)':<35} | {'Act_Z':<7}")
    print("-" * 115)

    for i, step in enumerate(traj):
        obs = step["obs"]
        acts = step["acts"]
        
        # 16D か 10D かでパースを変える
        if len(obs) == 16:
            z_mm = obs[9]
            fz = obs[[2, 5, 8]]
            # 角度を表示用に正規化
            angles = obs[11:16]
        else:
            # 10D (すでに正規化されている)
            # z_mm 復元: norm_z = (z / 16) + 1 => z = (norm_z - 1) * 16
            z_mm = (obs[3] - 1.0) * 16.0
            fz = obs[0:3]
            angles = obs[5:10]
        
        # ActionのZ方向
        act_z = acts[5]

        # 10ステップごとに区切り線（見やすくするため）
        if i > 0 and i % 10 == 0:
            print("-" * 115)

        # 指の角度 (AIが見ている正規化値) を整形
        ang_str = "[" + ", ".join([f"{a:4.1f}" for a in angles]) + "]"
        
        print(f"{i:4d} | {z_mm:6.1f} | {fz[0]:5.1f} {fz[1]:5.1f} {fz[2]:5.1f} | {obs[3]:6.2f} | {ang_str:<35} | {act_z:7.2f}")

    print("-" * 115)
    print("Dump Complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_dump_trajectory.py <filename.pkl>")
    else:
        # dataディレクトリ内のファイル名を直接指定できるように
        fname = sys.argv[1]
        if not fname.endswith(".pkl"):
            fname += ".pkl"
        dump_trajectory(fname)
