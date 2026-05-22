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
        radius = float(input("評価するボールの半径(mm)を入力してください: "))
        hw.ball_radius = radius
    except ValueError:
        hw.ball_radius = 0.0

    env = RobotHandEnv(hw)
    
    # モデルのロード
    if model_type == "bc":
        path = "models/bc_policy.pt"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
        policy = torch.load(path, weights_only=False)
        print("Using BC Policy.")
    elif model_type == "rl_ppo":
        path = "models/ppo_finetuned_model.zip"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
        model = PPO.load(path, env=env)
        policy = model.policy
        print("Using PPO Fine-tuned Policy.")
    elif model_type == "rl_sac":
        path = "models/sac_finetuned_model.zip"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
        model = SAC.load(path, env=env)
        policy = model.policy
        print("Using SAC Fine-tuned Policy.")
    else:
        print("Unknown model type.")
        return

    print("\n--- Evaluation Start ---")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    # 荷重統計用の記録リスト
    all_total_forces = []
    all_sensor_forces = [] # [[f1, f2, f3], ...]
    
    try:
        while not done:
            # 推論
            # 観測値は環境(env)側ですでに正しい10次元正規化済み
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = policy(obs_tensor)
            action = action_tensor.numpy()[0]
            
            # 実行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
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
