import os
import torch
import sys
import numpy as np
import time
from stable_baselines3 import PPO

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface
from envs.robot_hand_env import RobotHandEnv

import argparse

def evaluate(model_type="bc"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    # 既存の main.py からの引数渡しと input() の競合を避けるため、mode も引数で受け取れるようにする
    parser.add_argument("--mode", choices=["bc", "rl"])
    args, unknown = parser.parse_known_args()

    if args.mode:
        model_type = args.mode

    hw = HardwareInterface(dummy_mode=args.dummy)
    if not hw.connect():
        return

    try:
        radius = float(input("評価するボールの半径(mm)を入力してください: "))
        hw.set_ball_radius(radius)
    except ValueError:
        hw.set_ball_radius(0.0)

    env = RobotHandEnv(hw)
    
    # モデルのロード
    if model_type == "bc":
        path = "models/bc_policy.pt"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
        # PyTorch 2.6 以降のセキュリティ仕様に対応するため weights_only=False を指定
        policy = torch.load(path, weights_only=False)
        print("Using BC Policy.")
    else:
        path = "models/ppo_finetuned_model.zip"
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
        model = PPO.load(path)
        policy = model.policy
        print("Using PPO Fine-tuned Policy.")

    print("\n--- Evaluation Start ---")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    try:
        while not done:
            # 推論
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = policy(obs_tensor)
            action = action_tensor.numpy()[0]
            
            # 実行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # 状態表示
            step = info.get("step", 0)
            sys.stdout.write(f"\rStep: {step:3d}/200 ")
            hw.print_mms_status()
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    
    print(f"\nEvaluation Finished. Total Reward: {total_reward}, Success: {info.get('success', False)}")
    hw.disconnect()

if __name__ == "__main__":
    mode = input("評価するモデルを選択してください (bc/rl): ").lower()
    evaluate(mode)
