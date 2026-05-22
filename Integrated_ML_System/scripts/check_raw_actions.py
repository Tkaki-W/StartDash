import pickle
import os
import numpy as np

def check_raw():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    data_dir = os.path.join(base_dir, "data")
    
    # 最新のgraspファイルを1つ取得
    files = [f for f in os.listdir(data_dir) if "grasp" in f and f.endswith(".pkl")]
    if not files:
        print("No grasp files found.")
        return
    
    target = sorted(files)[-1]
    print(f"\nChecking raw actions in: {target}")

    with open(os.path.join(data_dir, target), 'rb') as f:
        data = pickle.load(f)
    
    traj = data[0] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) else data

    print(f"{'Step':<5} | {'Raw AI Actions (1-5)':<40}")
    print("-" * 60)
    # 最初、中間、最後をピックアップ
    indices = [0, len(traj)//2, len(traj)-1]
    for idx in indices:
        acts = traj[idx]["acts"]
        print(f"{idx:<5} | {acts}")

if __name__ == "__main__":
    check_raw()
