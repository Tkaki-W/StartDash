import os
import torch
import sys
from stable_baselines3 import SAC
import numpy as np

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
    
    try:
        radius = float(input("ボールの半径(mm)を入力してください: "))
        hw.set_ball_radius(radius)
    except ValueError:
        hw.set_ball_radius(0.0)

    env = RobotHandEnv(hw)

    # 2. SACモデルの作成
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    rl_model_path = "models/sac_finetuned_model.zip"
    bc_policy_path = "models/bc_policy.pt"

    if os.path.exists(rl_model_path):
        print(f"Loading existing SAC model from {rl_model_path} for continuation...")
        model = SAC.load(rl_model_path, env=env, learning_rate=3e-4)
    else:
        print("No existing SAC model found. Initializing...")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, 
                    policy_kwargs=policy_kwargs, 
                    buffer_size=10000, 
                    learning_starts=100)

        # BCポリシーのロードと重みの転送 (Actorへ)
        if os.path.exists(bc_policy_path):
            print(f"Loading BC policy from {bc_policy_path} as starting point...")
            bc_policy = torch.load(bc_policy_path, weights_only=False)

            # SACのActorに重みをコピー
            try:
                model.actor.features_extractor.load_state_dict(bc_policy.features_extractor.state_dict())
                model.actor.mlp_extractor.load_state_dict(bc_policy.mlp_extractor.state_dict())
                model.actor.mu.load_state_dict(bc_policy.action_net.state_dict())
                print("BC weights transferred to SAC Actor.")
            except Exception as e:
                print(f"Warning: Could not transfer all weights precisely: {e}")
        else:
            print("Warning: BC policy not found. Starting SAC from scratch.")


    # 4. 強化学習 (ファインチューン) の実行
    print("--- SAC RL Fine-tuning Start ---")
    try:
        # SACはサンプル効率が高いので、PPOより少なめでも効果が出やすいです
        model.learn(total_timesteps=1000) 
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 5. モデルの保存
    os.makedirs("models", exist_ok=True)
    model.save("models/sac_finetuned_model")
    print("Model saved to models/sac_finetuned_model")

    hw.disconnect()

if __name__ == "__main__":
    train()
