import pickle
import os
import numpy as np
import sys

def check_data(filename=None):
    data_dir = "data"
    
    if filename:
        file_path = os.path.join(data_dir, filename)
    else:
        # 最新のファイルを探す
        files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        if not files:
            print("エラー: dataフォルダに .pkl ファイルが見つかりません。")
            return
        files.sort()
        file_path = os.path.join(data_dir, files[-1])

    print(f"\n" + "="*50)
    print(f" 読み込み中: {file_path}")
    print("="*50)

    try:
        with open(file_path, 'rb') as f:
            trajectories = pickle.load(f)
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました。 {e}")
        return

    # データの構造チェック
    # 通常は [ [step1, step2, ...], ... ] というリストのリスト形式
    num_episodes = len(trajectories)
    print(f"合計エピソード数: {num_episodes}")

    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        print(f"\n[エピソード {i+1}] 合計ステップ数: {num_steps}")
        
        if num_steps > 0:
            # 最初のステップを表示
            first_step = traj[0]
            obs = first_step["obs"]
            acts = first_step["acts"]
            
            print(f" --- Start Step (0) ---")
            print(f"  Obs (16次元): {np.round(obs, 3)}")
            print(f"    - CNC Z: {obs[9]:.2f}")
            print(f"    - Radius: {obs[10]:.2f}")
            print(f"  Acts (6次元): {np.round(acts, 3)}")
            print(f"    - Target Z (Acts[5]): {acts[5]:.3f}")

            # 最後のステップを表示
            if num_steps > 1:
                last_step = traj[-1]
                obs_l = last_step["obs"]
                acts_l = last_step["acts"]
                print(f" --- Last Step ({num_steps-1}) ---")
                print(f"  Obs (16次元): {np.round(obs_l, 3)}")
                print(f"    - CNC Z: {obs_l[9]:.2f}")
                print(f"    - Radius: {obs_l[10]:.2f}")
                print(f"  Acts (6次元): {np.round(acts_l, 3)}")
                print(f"    - Target Z (Acts[5]): {acts_l[5]:.3f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    # 引数があればそのファイル、なければ最新のものを読み込む
    target = sys.argv[1] if len(sys.argv) > 1 else None
    check_data(target)
