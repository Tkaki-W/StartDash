import pickle
import os
import numpy as np

def inspect_all():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    files.sort()

    print(f"{'Filename':<35} | {'Steps':<5} | {'Start Z':<7} | {'End Z':<7} | {'End Angles'}")
    print("-" * 90)

    for f in files:
        try:
            with open(os.path.join(data_dir, f), 'rb') as pkl:
                data = pickle.load(pkl)
            
            # データの形式判定
            traj = data[0] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) else data
            
            if len(traj) == 0: continue

            start_z = traj[0]["obs"][9]
            end_z = traj[-1]["obs"][9]
            # 角度 (90±80) に復元
            end_angles = [int(round(a * 80 + 90)) for a in traj[-1]["obs"][11:16]]
            
            print(f"{f:<35} | {len(traj):<5} | {start_z:<7.1f} | {end_z:<7.1f} | {end_angles}")
        except Exception as e:
            print(f"{f:<35} | Error reading file: {e}")

if __name__ == "__main__":
    inspect_all()
