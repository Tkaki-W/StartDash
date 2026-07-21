import os
import pickle
import numpy as np
import gymnasium as gym
from imitation.algorithms import bc
from imitation.data import types, rollout
import torch
import sys
import argparse
import torch.nn as nn

def load_expert_data(data_path):
    with open(data_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    if isinstance(trajectories_data, list) and len(trajectories_data) > 0 and not isinstance(trajectories_data[0], list):
        trajectories_data = [trajectories_data]
        
    trajectories = []
    for traj_data in trajectories_data:
        raw_obs = np.array([d["obs"] for d in traj_data], dtype=np.float32)
        raw_acts = np.array([d["acts"] for d in traj_data], dtype=np.float32)
        
        if raw_obs.shape[1] != 10:
            continue
            
        # [Fz1, Fz2, Fz3, Radius, Ang1, Ang2, Ang3, Ang4, Ang5]
        obs_9d = np.concatenate([raw_obs[:, 0:3], raw_obs[:, 4:10]], axis=1)

        # アクション (5次元の角度)
        acts_5d = raw_acts[:, 0:5]

        # Trajectory作成 (フィルタリングなし、全ステップを使用)
        if len(acts_5d) < 2: continue
        final_obs = obs_9d[-1:]
        obs_aug = np.concatenate([obs_9d, final_obs], axis=0)
        infos = [{} for _ in range(len(acts_5d))]
        trajectories.append(types.Trajectory(obs=obs_aug, acts=acts_5d, infos=infos, terminal=True))
    
    return trajectories

# REACH専用
class ReachRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["reach", "grasp"], default="grasp")
    args = parser.parse_args()
    
    phase = args.phase
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "data")
    
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl") and f"_{phase}_" in f]
    if not files: return

    if phase == "reach":
        print(f"--- Training Reach Regressor ---")
        X, Y = [], []
        for f in files:
            try:
                with open(f, 'rb') as pkl:
                    traj = pickle.load(pkl); obs = traj[0]["obs"]
                    if len(obs) == 1: X.append([obs[0]]); Y.append([traj[-1]["acts"][0]])
            except: continue
        if not X: return
        model = ReachRegressor(); optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss(); X_t = torch.tensor(X, dtype=torch.float32); Y_t = torch.tensor(Y, dtype=torch.float32)
        for epoch in range(1001):
            optimizer.zero_grad(); loss = criterion(model(X_t), Y_t); loss.backward(); optimizer.step()
        os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
        torch.save(model, os.path.join(base_dir, "models", "reach_model.pt"))
        print("Reach model saved.")
    else:
        print(f"--- Training Grasp Expert (Tactile-to-Angle Mode) ---")
        all_trajectories = []
        for f in files: all_trajectories.extend(load_expert_data(f))
        if not all_trajectories:
            print("Error: No stable contact found in data files.")
            return
            
        transitions = rollout.flatten_trajectories(all_trajectories)
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        from stable_baselines3.common.policies import ActorCriticPolicy
        def constant_lr(_): return 0.001
        custom_policy = ActorCriticPolicy(
            observation_space=observation_space, action_space=action_space,
            lr_schedule=constant_lr, net_arch=dict(pi=[32, 32], vf=[32, 32])
        )
        bc_trainer = bc.BC(
            observation_space=observation_space, action_space=action_space,
            demonstrations=transitions, batch_size=64, policy=custom_policy,
            rng=np.random.default_rng(42)
        )
        bc_trainer.train(n_epochs=500)
        os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
        torch.save(bc_trainer.policy, os.path.join(base_dir, "models", "bc_grasp_policy.pt"))
        print("Grasp model saved.")

if __name__ == "__main__":
    train()
