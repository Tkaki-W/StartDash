import os
import pickle
import numpy as np
import gymnasium as gym
from imitation.algorithms import bc
from imitation.data import types
import torch
import sys

from imitation.data import types, rollout

def load_expert_data(data_path):
    with open(data_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    # 古い形式（1つの長いリスト）の場合はリストで包む
    if isinstance(trajectories_data, list) and len(trajectories_data) > 0 and not isinstance(trajectories_data[0], list):
        trajectories_data = [trajectories_data]
        
    trajectories = []
    for traj_data in trajectories_data:
        obs = np.array([d["obs"] for d in traj_data], dtype=np.float32)
        acts = np.array([d["acts"] for d in traj_data], dtype=np.float32)
        
        # 最後の観測値を補完 (Trajectoryの仕様上、len(obs) == len(acts) + 1 が必要)
        final_obs = obs[-1:]
        obs = np.concatenate([obs, final_obs], axis=0)
        infos = [{} for _ in range(len(acts))]
        
        trajectories.append(types.Trajectory(obs=obs, acts=acts, infos=infos, terminal=True))
    
    return trajectories

def train():
    data_dir = "data"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("expert_demo_")]
    if not files:
        print("エラー: 訓練データが見つかりません。")
        return
    
    all_trajectories = []
    for f in files:
        print(f"Loading expert data: {f}")
        all_trajectories.extend(load_expert_data(f))
    
    print(f"Total Trajectories: {len(all_trajectories)}")
    transitions = rollout.flatten_trajectories(all_trajectories)
    
    # 観測 16次元, アクション 6次元
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        batch_size=32,
        rng=np.random.default_rng(42) # 乱数シードを固定
    )

    print("--- BC Training Start ---")
    bc_trainer.train(n_epochs=500)
    
    os.makedirs("models", exist_ok=True)
    torch.save(bc_trainer.policy, "models/bc_policy.pt")
    print("Policy saved to models/bc_policy.pt")

if __name__ == "__main__":
    train()
