import os
import torch
import sys
import numpy as np
import time
from stable_baselines3 import PPO, SAC

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface
from envs.robot_hand_env import RobotHandEnv

import argparse

# 1. REACHモデルの設計図 (保存されたモデルを読み込むために必要)
class ReachRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

def evaluate(model_type="bc"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    # 既存の main.py からの引数渡しと input() の競合を避けるため、mode も引数で受け取れるようにする
    parser.add_argument("--mode", choices=["bc", "rl_ppo", "rl_sac"])
    args, unknown = parser.parse_known_args()

    if args.mode:
        model_type = args.mode

    hw = HardwareInterface(dummy_mode=args.dummy)
    if not hw.connect():
        return

    try:
        radius = float(input("評価する物体のサイズ(mm)を入力してください: "))
        hw.ball_radius = radius
    except ValueError:
        hw.ball_radius = 0.0

    shape = input("物体の形状 (ball / cube): ").lower()
    if shape not in ['ball', 'cube']:
        print("Error: 'ball' または 'cube' を入力してください。")
        hw.disconnect()
        return

    # 1. REACHモデルのロード (半径 -> 高度)
    reach_path = f"models/reach_model_{shape}.pt"
    reach_model = None
    if os.path.exists(reach_path):
        # クラス定義は冒頭に移動済み
        reach_model = torch.load(reach_path, weights_only=False)
        print(f"Loaded Reach Regressor for {shape}.")
    else:
        print(f"Warning: {reach_path} not found. Using default reach height.")

    # 2. 環境の初期化 (REACHモデルを渡す)
    env = RobotHandEnv(hw, reach_model=reach_model)
    
    # 3. 把持ポリシーのロード (9次元)
    policy = None
    if model_type == "bc":
        path = f"models/bc_grasp_policy_{shape}.pt"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            hw.disconnect()
            return
        policy = torch.load(path, weights_only=False)
        print(f"Using BC Grasp Policy for {shape} (9D).")
    elif model_type == "rl_ppo":
        path = f"models/ppo_finetuned_model_{shape}.zip"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            hw.disconnect()
            return
        model = PPO.load(path, env=env)
        policy = model.policy
        print(f"Using PPO Fine-tuned Policy for {shape} (9D).")
    elif model_type == "rl_sac":
        path = f"models/sac_finetuned_model_{shape}.zip"
        if not os.path.exists(path):
            # .zip 拡張子がない可能性も考慮
            path_no_ext = f"models/sac_finetuned_model_{shape}"
            if os.path.exists(path_no_ext + ".zip"):
                path = path_no_ext + ".zip"
            elif os.path.exists(path_no_ext):
                path = path_no_ext
            else:
                print(f"Error: {path} not found.")
                hw.disconnect()
                return
        model = SAC.load(path, env=env)
        policy = model.policy
        print(f"Using SAC Fine-tuned Policy for {shape} (9D).")
    else:
        print("Unknown model type.")
        hw.disconnect()
        return

    print("\n--- Evaluation Start (Hierarchical: Reach -> Grasp) ---")
    obs, _ = env.reset() # ここで自動降下が行われる
    done = False
    total_reward = 0
    
    # 荷重統計用の記録リスト
    all_total_forces = []
    all_sensor_forces = [] 
    
    try:
        while not done:
            # 観測値を把持専門の9次元に確実に変換
            if len(obs) == 10:
                # [Fz1, Fz2, Fz3, Z, Radius, A1-5] から Z(idx 3) を削除
                obs_9d = np.concatenate([obs[0:3], obs[4:10]])
            else:
                obs_9d = obs

            # 【デバッグ】AIへの入力を表示 (Fz*3, Radius, Angles*5)
            print(f"\rAI Obs: Fz[{obs_9d[0]:.1f},{obs_9d[1]:.1f},{obs_9d[2]:.1f}] R:{obs_9d[3]:.2f} Angs:{obs_9d[4:9]}", end="")

            # 確定的なアクションを取得 (評価のため deterministic=True に設定)
            action, _ = policy.predict(obs_9d, deterministic=True)
            
            # 実行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # 【デバッグ】AIの予測アクションを表示 (+1.0が閉じ)
            if info.get("step", 0) % 20 == 0:
                print(f"\nAI Action Predict: {action} (Target: All +1.0)")
            
            # 荷重の記録 (infoから取得)
            total_f = info.get("total_force", 0.0)
            sensor_fs = info.get("forces", [0.0]*3)
            all_total_forces.append(total_f)
            all_sensor_forces.append(sensor_fs)
            
            # 状態表示
            step = info.get("step", 0)
            prefix = f"Step: {step:3d}/500 | Total Force: {total_f:5.2f}"
            hw.print_mms_status(prefix=prefix)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    
    # 統計計算
    if all_total_forces:
        avg_f = np.mean(all_total_forces)
        max_f = np.max(all_total_forces)
        max_per_sensor = np.max(np.abs(all_sensor_forces), axis=0)
    else:
        avg_f = max_f = 0.0
        max_per_sensor = [0.0]*3

    print(f"\n\n--- Evaluation Results ---")
    print(f" Success: {info.get('success', False)}")
    print(f" Total Reward: {total_reward:.2f}")
    print(f" Average Total Force: {avg_f:.2f} N")
    print(f" Maximum Total Force: {max_f:.2f} N")
    print(f" Peak Force per Finger: S1:{max_per_sensor[0]:.2f}N, S2:{max_per_sensor[1]:.2f}N, S3:{max_per_sensor[2]:.2f}N")
    print("---------------------------\n")
    
    hw.disconnect()

if __name__ == "__main__":
    evaluate()
