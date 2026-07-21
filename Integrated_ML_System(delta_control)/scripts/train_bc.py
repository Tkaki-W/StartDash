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
        # raw_obs: [Fz1, Fz2, Fz3, Radius, Ang1, Ang2, Ang3, Ang4, Ang5] (9次元)
        raw_obs = np.array([d["obs"] for d in traj_data], dtype=np.float32)
        # raw_acts: [Ang1, Ang2, Ang3, Ang4, Ang5] (正規化済み)
        raw_acts = np.array([d["acts"] for d in traj_data], dtype=np.float32)
        
        if raw_obs.shape[1] != 9:
            continue

        # --- Delta Control への変換 (正規化を一回のみ) ---
        raw_deltas = np.diff(raw_acts, axis=0)
        raw_deltas = np.concatenate([raw_deltas, np.zeros((1, 5))], axis=0)
        
        norm_deltas = np.zeros_like(raw_deltas)
        norm_deltas[:, 0] = raw_deltas[:, 0] * (75.0 / 5.0)       # 親指: (Δa * 75.0) / 5.0
        norm_deltas[:, 1:] = raw_deltas[:, 1:] * -(80.0 / 5.0)    # その他: (-Δa * 80.0) / 5.0
        norm_deltas = np.clip(norm_deltas, -1.0, 1.0)

        # Trajectory作成
        if len(norm_deltas) < 2: continue
        # Trajectoryクラスの仕様に合わせて、obsの末尾にダミー（最後と同じ状態）を1つ足す
        final_obs = raw_obs[-1:]
        obs_aug = np.concatenate([raw_obs, final_obs], axis=0)
        infos = [{} for _ in range(len(norm_deltas))]
        trajectories.append(types.Trajectory(obs=obs_aug, acts=norm_deltas, infos=infos, terminal=True))
    
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

    # 形状の選択
    shape = input("学習する物体の形状 (ball / cube): ").lower()
    if shape not in ['ball', 'cube']:
        print("Error: 'ball' または 'cube' を入力してください。")
        return

    phase = args.phase
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "data")

    # 指定された形状のファイルのみを読み込む
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.endswith(".pkl") and f"_{phase}_" in f and f"_{shape}_" in f]
    if not files:
        print(f"Error: {shape}の{phase}データが見つかりません。")
        return

    if phase == "reach":
        print(f"--- Training Reach Regressor ---")
        X, Y = [], []
        for f in files:
            try:
                with open(f, 'rb') as pkl:
                    traj = pickle.load(pkl)
                    reach_traj = traj[0]
                    obs = reach_traj[0]["obs"]
                    if len(obs) == 1:
                        X.append([obs[0]])
                        Y.append([reach_traj[-1]["acts"][0]])
            except: continue
        if not X: return
        model = ReachRegressor(); optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss(); X_t = torch.tensor(X, dtype=torch.float32); Y_t = torch.tensor(Y, dtype=torch.float32)
        for epoch in range(1001):
            optimizer.zero_grad(); loss = criterion(model(X_t), Y_t); loss.backward(); optimizer.step()
        os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
        model_path = os.path.join(base_dir, "models", f"reach_model_{shape}.pt")
        torch.save(model, model_path)
        print(f"Reach model saved to {model_path}")
    else:
        print(f"--- Training Grasp Expert (Tactile-to-Angle Mode) ---")
        all_trajectories = []
        for f in files: all_trajectories.extend(load_expert_data(f))
        if not all_trajectories:
            print("Error: No trajectory loaded from data files.")
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
        model_path = os.path.join(base_dir, "models", f"bc_grasp_policy_{shape}.pt")
        torch.save(bc_trainer.policy, model_path)
        print(f"Grasp model saved to {model_path}")

if __name__ == "__main__":
    train()
