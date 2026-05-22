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
    
    # ボールの属性を設定
    try:
        radius = float(input("ボールの半径(mm)を入力してください: "))
        hw.ball_radius = radius
    except ValueError:
        radius = 0.0
        hw.ball_radius = 0.0

    hardness = input("ボールの硬さ (soft / hard): ").lower()
    if hardness not in ['soft', 'hard']:
        hardness = "unknown"

    env = RobotHandEnv(hw)
    env = DummyVecEnv([lambda: env])

    # 2. PPOモデルの作成
    # 模倣学習の重みを引き継ぐため、同じネットワーク構造にする必要があります
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)

    # 3. BCポリシーのロードと重みの転送
    bc_policy_path = "models/bc_policy.pt"

    if os.path.exists(rl_model_path):
        print(f"Loading existing RL model from {rl_model_path} for continuation...")
        model = PPO.load(rl_model_path, env=env, learning_rate=1e-4)
    else:
        print("No existing RL model found. Initializing...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, 
                    n_steps=2048, policy_kwargs=policy_kwargs)

        # BCポリシーのロードと重みの転送
        if os.path.exists(bc_policy_path):
            print(f"Loading BC policy from {bc_policy_path} as starting point...")
            bc_policy = torch.load(bc_policy_path, weights_only=False)
            model.policy.load_state_dict(bc_policy.state_dict())
            print("BC weights transferred to PPO model.")
        else:
            print("Warning: BC policy not found. Starting RL from scratch.")

    # 4. 強化学習 (ファインチューン) の実行
    print(f"--- RL Fine-tuning Start (Object: {radius}mm {hardness}) ---")
    
    import csv
    from datetime import datetime
    from stable_baselines3.common.callbacks import BaseCallback

    class SaveAndLogCallback(BaseCallback):
        def __init__(self, save_path, radius, hardness, verbose=0):
            super(SaveAndLogCallback, self).__init__(verbose)
            self.save_path = save_path
            self.radius = radius
            self.hardness = hardness
            self.log_file = "logs/rl_history.csv"
            
            # ログファイルのヘッダー作成
            if not os.path.exists(self.log_file):
                os.makedirs("logs", exist_ok=True)
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "radius", "hardness", "reward", "success"])

        def _on_step(self) -> bool:
            if self.locals["dones"][0]:
                info = self.locals["infos"][0]
                reward = self.locals["rewards"][0]
                success = info.get("success", False)
                
                # CSVに記録
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.radius,
                        self.hardness,
                        round(float(reward), 3),
                        1 if success else 0
                    ])

                print(f"\n[Safety Save & Log] Episode finished. Reward: {reward:.2f}, Success: {success}")
                self.model.save(self.save_path)
            return True

    save_callback = SaveAndLogCallback(rl_model_path, radius, hardness)

    try:
        model.learn(total_timesteps=10000, callback=save_callback) 
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 最終保存
    os.makedirs("models", exist_ok=True)
    model.save(rl_model_path)
    print(f"Final model saved to {rl_model_path}")

    hw.disconnect()

if __name__ == "__main__":
    train()
