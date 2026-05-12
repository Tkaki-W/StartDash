import os
import torch
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface
from envs.robot_hand_env import RobotHandEnv

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    # 1. ハードウェアと環境の準備
    hw = HardwareInterface(dummy_mode=args.dummy)
    if not hw.connect():
        print("Error: Could not connect to hardware.")
        return
    
    # ボールの半径を設定
    try:
        radius = float(input("ボールの半径(mm)を入力してください: "))
        hw.set_ball_radius(radius)
    except ValueError:
        hw.set_ball_radius(0.0)

    env = RobotHandEnv(hw)
    env = DummyVecEnv([lambda: env])

    # 2. PPOモデルの作成
    # 模倣学習の重みを引き継ぐため、同じネットワーク構造にする必要があります
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)

    # 3. BCポリシーのロードと重みの転送
    bc_policy_path = "models/bc_policy.pt"
    if os.path.exists(bc_policy_path):
        print(f"Loading BC policy from {bc_policy_path}...")
        # PyTorch 2.6 以降のセキュリティ仕様に対応するため weights_only=False を指定
        bc_policy = torch.load(bc_policy_path, weights_only=False)
        
        # SB3のPPOモデルのポリシーに重みをコピー
        model.policy.load_state_dict(bc_policy.state_dict())
        print("BC weights transferred to PPO model.")
    else:
        print("Warning: BC policy not found. Starting RL from scratch.")

    # 4. 強化学習 (ファインチューン) の実行
    print("--- RL Fine-tuning Start ---")
    try:
        model.learn(total_timesteps=1000) # 実機なので少なめに設定
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 5. モデルの保存
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_finetuned_model")
    print("Model saved to models/ppo_finetuned_model")

    hw.disconnect()

if __name__ == "__main__":
    train()
