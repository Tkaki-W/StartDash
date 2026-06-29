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

            obs = traj[0]["obs"]
            obs_dim = len(obs)
            
            # 次元数に応じてインデックスを切り替え
            if obs_dim == 16:
                # 古い16次元データ
                start_z = obs[9]
                end_z = traj[-1]["obs"][9]
                end_angles = [int(round(a * 80 + 90)) for a in traj[-1]["obs"][11:16]]
            elif obs_dim == 10:
                # 新しい10次元データ
                # Zの復元: norm_z = (z / 16) + 1 => z = (norm_z - 1) * 16
                start_z = (obs[3] - 1.0) * 16.0
                end_z = (traj[-1]["obs"][3] - 1.0) * 16.0
                # 角度の復元 (簡易的に表示)
                # 親指(idx 5): (raw-95)/75, 他(idx 6-9): -(raw-100)/80
                raw_0 = int(round(traj[-1]["obs"][5] * 75.0 + 95.0))
                raw_others = [int(round(100.0 - (a * 80.0))) for a in traj[-1]["obs"][6:10]]
                end_angles = [raw_0] + raw_others
            else:
                start_z = 0; end_z = 0; end_angles = ["Unknown Dim"]

            print(f"{f:<35} | {len(traj):<5} | {start_z:<7.1f} | {end_z:<7.1f} | {end_angles} ({obs_dim}D)")
        except Exception as e:
            print(f"{f:<35} | Error reading file: {e}")

if __name__ == "__main__":
    inspect_all()
