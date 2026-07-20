import os
import torch
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import csv
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface
from envs.robot_hand_env import RobotHandEnv
from scripts.train_bc import ReachRegressor

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    hw = HardwareInterface(dummy_mode=args.dummy)
    if not hw.connect(): return
    
    try:
        radius = float(input("ボールの半径(mm)を入力してください: "))
        hw.ball_radius = radius
    except ValueError:
        radius = 0.0; hw.ball_radius = 0.0

    hardness = input("ボールの硬さ (soft / hard): ").lower()

    # 1. モデルの準備
    reach_path = "models/reach_model.pt"
    reach_model = None
    if os.path.exists(reach_path):
        reach_model = torch.load(reach_path, weights_only=False)
        print("Loaded Reach Regressor.")

    env = RobotHandEnv(hw, reach_model=reach_model)
    env = DummyVecEnv([lambda: env])

    # 2. PPOモデルの作成 (9次元入力, 5次元出力)
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    rl_model_path = "models/ppo_finetuned_model.zip"
    bc_policy_path = "models/bc_grasp_policy.pt"

    if os.path.exists(rl_model_path):
        print(f"Loading existing RL model from {rl_model_path}...")
        model = PPO.load(rl_model_path, env=env, learning_rate=1e-4)
    else:
        print("No existing RL model found. Initializing from BC...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, 
                    n_steps=1024, batch_size=64, policy_kwargs=policy_kwargs)

        if os.path.exists(bc_policy_path):
            bc_policy = torch.load(bc_policy_path, weights_only=False)
            model.policy.load_state_dict(bc_policy.state_dict())
            print("BC weights transferred to PPO model.")

    # 3. コールバックと学習
    class SaveAndLogCallback(BaseCallback):
        def __init__(self, save_path, radius, hardness, verbose=0):
            super(SaveAndLogCallback, self).__init__(verbose)
            self.save_path = save_path
            self.radius = radius
            self.hardness = hardness
            self.log_file = "logs/rl_history.csv"
            self.episode_reward = 0.0 # 累積報酬用のカウンタ
            if not os.path.exists("logs"): os.makedirs("logs", exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(["timestamp", "radius", "hardness", "total_reward", "success"])

        def _on_step(self) -> bool:
            # 毎ステップの報酬を加算
            self.episode_reward += self.locals["rewards"][0]

            if self.locals["dones"][0]:
                info = self.locals["infos"][0]
                success = info.get("success", False)
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        self.radius, 
                        self.hardness, 
                        round(float(self.episode_reward), 3), 
                        1 if success else 0
                    ])
                print(f"\n[Safety Save] Total Reward: {self.episode_reward:.2f}, Success: {success}")
                self.model.save(self.save_path)
                # 次のエピソードのためにリセット
                self.episode_reward = 0.0
            return True

    print(f"--- RL Fine-tuning Start ---")
    try:
        model.learn(total_timesteps=10000, callback=SaveAndLogCallback(rl_model_path, radius, hardness))
    except KeyboardInterrupt:
        print("Interrupted.")
    
    model.save(rl_model_path)
    print(f"Final model saved to {rl_model_path}")
    hw.disconnect()

if __name__ == "__main__":
    train()
