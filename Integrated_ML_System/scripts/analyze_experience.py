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
    expert_stats = defaultdict(int)

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
            
            expert_stats[label] += 1

    print(f"\n[1] 模倣学習用データ (エキスパート実演)")
    if not expert_stats:
        print("  データが見つかりません。")
    else:
        for label, count in sorted(expert_stats.items()):
            print(f"  - {label}: {count} 件")

    # 2. 強化学習履歴 (RL History) の集計
    log_file = "logs/rl_history.csv"
    rl_stats = defaultdict(lambda: {"episodes": 0, "successes": 0, "total_reward": 0.0})

    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = f"{int(float(row['radius']))}mm_{row['hardness']}"
                rl_stats[label]["episodes"] += 1
                rl_stats[label]["successes"] += int(row["success"])
                rl_stats[label]["total_reward"] += float(row["reward"])

    print(f"\n[2] 強化学習の経験 (自己試行錯誤)")
    if not rl_stats:
        print("  履歴が見つかりません。")
    else:
        for label, stats in rl_stats.items():
            succ_rate = (stats["successes"] / stats["episodes"]) * 100
            avg_rew = stats["total_reward"] / stats["episodes"]
            print(f"  - {label.replace('_', ' ')}:")
            print(f"      試行回数: {stats['episodes']} エピソード")
            print(f"      成功率  : {succ_rate:.1f}% ({stats['successes']}/{stats['episodes']})")
            print(f"      平均報酬: {avg_rew:.2f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    analyze()
