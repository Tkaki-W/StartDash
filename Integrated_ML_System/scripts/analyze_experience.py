import os
import csv
import pickle
import numpy as np
from collections import defaultdict

def analyze():
    # スクリプトの場所を基準に、正しいディレクトリを特定する
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    data_dir = os.path.join(base_dir, "data")
    log_file = os.path.join(base_dir, "logs", "rl_history.csv")

    print("="*50)
    print("      5本指ロボットハンド 学習経験 分析レポート")
    print(f"      参照先: {base_dir}")
    print("="*50)

    # 1. 模倣学習データ (Expert Data) の集計
    expert_file_counts = defaultdict(int)
    expert_step_counts = defaultdict(int)

    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        for f in files:
            # ファイル名から情報を抽出 (例: 4mm_hard_2026...)
            parts = f.split('_')
            
            if len(parts) >= 2:
                # 最初の要素がサイズ(4mmなど)、二番目が硬さ(hardなど)
                label = f"{parts[0]} {parts[1]}"
            else:
                label = "other format"
            
            expert_file_counts[label] += 1
            
            # ステップ数を計測
            try:
                with open(os.path.join(data_dir, f), 'rb') as pkl:
                    traj = pickle.load(pkl)
                    # データの形式（リストのリストか単一リストか）を判定
                    steps = len(traj[0]) if isinstance(traj[0], list) and len(traj) > 0 else len(traj)
                    expert_step_counts[label] += steps
            except:
                pass

    print(f"\n[1] 模倣学習用データ (エキスパート実演)")
    if not expert_file_counts:
        print("  データが見つかりません。")
    else:
        for label in sorted(expert_file_counts.keys()):
            count = expert_file_counts[label]
            total_steps = expert_step_counts[label]
            avg_steps = total_steps / count if count > 0 else 0
            print(f"  - {label}: {count} 件 (平均 {int(avg_steps)} ステップ/件)")

    # 2. 強化学習履歴 (RL History) の集計
    rl_stats = defaultdict(lambda: {"episodes": 0, "successes": 0, "total_reward": 0.0})

    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = f"{int(float(row['radius']))}mm {row['hardness']}"
                rl_stats[label]["episodes"] += 1
                rl_stats[label]["successes"] += int(row["success"])
                rl_stats[label]["total_reward"] += float(row["reward"])

    print(f"\n[2] 強化学習の経験 (自己試行錯誤)")
    if not rl_stats:
        print("  履歴が見つかりません (logs/rl_history.csv なし)")
    else:
        for label, stats in sorted(rl_stats.items()):
            succ_rate = (stats["successes"] / stats["episodes"]) * 100
            avg_rew = stats["total_reward"] / stats["episodes"]
            print(f"  - {label}:")
            print(f"      試行回数: {stats['episodes']} エピソード")
            print(f"      成功率  : {succ_rate:.1f}% ({stats['successes']}/{stats['episodes']})")
            print(f"      平均報酬: {avg_rew:.2f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    analyze()
