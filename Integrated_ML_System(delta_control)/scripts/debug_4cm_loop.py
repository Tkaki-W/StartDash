import os
import torch
import sys
import numpy as np
import time
from stable_baselines3 import SAC

# 自作モジュールのインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.hardware_interface import HardwareInterface
from envs.robot_hand_env import RobotHandEnv

def debug_loop():
    # ダミーモードではなく実機で確認することを推奨しますが、
    # ユーザーの環境に合わせて HardwareInterface を初期化します
    hw = HardwareInterface(dummy_mode=False) 
    if not hw.connect():
        print("Hardware connection failed.")
        return

    # 4cm (40mm) に固定
    radius = 4.0
    hw.ball_radius = radius
    print(f"Target Radius: {radius}mm")

    # モデルのロード
    policy_path = "models/bc_grasp_policy.pt"
    if not os.path.exists(policy_path):
        print(f"Model {policy_path} not found.")
        hw.disconnect()
        return
    policy = torch.load(policy_path, weights_only=False)
    
    # 環境
    env = RobotHandEnv(hw)
    obs, _ = env.reset()
    
    print("\n--- Debug Logging Start ---")
    print("Step | Fz1  Fz2  Fz3 | Ang1  Ang2  Ang3  Ang4  Ang5 | Act1  Act2  Act3  Act4  Act5")
    print("-" * 85)

    try:
        for step in range(1, 1001):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = policy(obs_tensor)
            action = action_tensor.numpy()[0]
            
            # ログ表示 (生の値に近い状態で表示)
            fz = obs[0:3]
            angs = obs[4:9]
            
            print(f"{step:4d} | {fz[0]:.1f} {fz[1]:.1f} {fz[2]:.1f} | {angs[0]:.2f} {angs[1]:.2f} {angs[2]:.2f} {angs[3]:.2f} {angs[4]:.2f} | {action[0]:.2f} {action[1]:.2f} {action[2]:.2f} {action[3]:.2f} {action[4]:.2f}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                print("\nEpisode finished.")
                break
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hw.disconnect()

if __name__ == "__main__":
    debug_loop()
