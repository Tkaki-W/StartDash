import pickle
import os
import sys
import numpy as np

def inspect(filename):
    # スクリプトの場所を基準にパスを設定
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    file_path = os.path.join(base_dir, "data", filename)
    
    if not os.path.exists(file_path):
        print(f"エラー: {file_path} が見つかりません。")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"読み込み失敗: {e}")
        return
    
    # データの形式（リストのリストか単一リストか）を判定
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        traj = data[0]
    else:
        traj = data

    print(f"\nファイル名: {filename}")
    print(f"合計ステップ数: {len(traj)}")
    print("-" * 75)
    print(f"{'Step':<5} | {'Time(s)':<8} | {'CNC Z':<8} | {'Fingers (1-5) [Deg]':<30}")
    print("-" * 75)

    for i, step in enumerate(traj):
        obs = step["obs"]
        
        # 記録頻度 20Hz = 0.05秒 ごとの推定時間
        time_est = i * 0.05
        cnc_z = obs[9]
        
        # 観測値の角度 (11-15) を正規化から度数(90±80)に復元
        # norm_angles = (angles - 90.0) / 80.0 だったので:
        angles = [int(round(a * 80 + 90)) for a in obs[11:16]]
        
        # 全件表示だと長すぎる場合があるので、適宜調整可能ですが
        # 研究用なのでまずは全件出力します（必要なら head/tail などで確認してください）
        print(f"{i:<5} | {time_est:<8.2f} | {cnc_z:<8.2f} | {angles}")

    print("-" * 75)
    print("解析完了。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python scripts/inspect_trajectory.py <ファイル名.pkl>")
    else:
        inspect(sys.argv[1])
