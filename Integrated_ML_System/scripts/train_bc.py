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
        raw_obs = np.array([d["obs"] for d in traj_data], dtype=np.float32)
        raw_acts = np.array([d["acts"] for d in traj_data], dtype=np.float32)

        # 既存の16次元データから10次元へ抽出・変換
        if raw_obs.shape[1] == 16:
            obs_10d = np.zeros((raw_obs.shape[0], 10), dtype=np.float32)

            # 1. Fz のドリフトガード (プラスを0に)
            obs_10d[:, 0:3] = np.minimum(0.0, raw_obs[:, [2, 5, 8]])

            # 2. CNC Z の正規化 (raw Z=0.0 -> +1.0, Z=-32.0 -> -1.0)
            obs_10d[:, 3] = (raw_obs[:, 9] / 16.0) + 1.0

            # 3. ボール半径の正規化 (10mmを1.0とする)
            obs_10d[:, 4] = raw_obs[:, 10] / 10.0

            # 4. 指の角度の正規化を「+1.0=閉」に統一
            # 旧データの正規化: old_norm = (angle - 90) / 80.0
            # 親指 (idx 5): 中心95, スケール75 
            obs_10d[:, 5] = ( (raw_obs[:, 11] * 80.0 + 90.0) - 95.0 ) / 75.0
            # 他の指 (idx 6-9): 中心100, 反転, スケール80
            for i in range(1, 5):
                obs_10d[:, 5 + i] = -( (raw_obs[:, 11 + i] * 80.0 + 90.0) - 100.0 ) / 80.0

            obs = obs_10d

            # 出力(acts)も同様に修正 (acts idx 0-4)
            acts = raw_acts.copy()
            # 親指 (idx 0)
            acts[:, 0] = ( (raw_acts[:, 0] * 80.0 + 90.0) - 95.0 ) / 75.0
            # 他の指 (idx 1-4)
            acts[:, 1:5] = -( (raw_acts[:, 1:5] * 80.0 + 90.0) - 100.0 ) / 80.0
        else:
            obs = raw_obs
            acts = raw_acts

        # 最後の観測値を補完


        final_obs = obs[-1:]
        obs = np.concatenate([obs, final_obs], axis=0)
        infos = [{} for _ in range(len(acts))]
        
        trajectories.append(types.Trajectory(obs=obs, acts=acts, infos=infos, terminal=True))
    
    return trajectories

def train():
    data_dir = "data"
    # 新しい形式 (3mm_soft_...) と旧形式 (expert_demo_...) の両方に対応するため、.pkl ファイルをすべて取得
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]
    if not files:
        print("エラー: 訓練データ (.pkl) が見つかりません。")
        return
    
    all_trajectories = []
    for f in files:
        print(f"Loading expert data: {f}")
        all_trajectories.extend(load_expert_data(f))
    
    print(f"Total Trajectories: {len(all_trajectories)}")
    transitions = rollout.flatten_trajectories(all_trajectories)
    
    # 観測 10次元, アクション 6次元
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    # ネットワーク構造 [256, 256] のポリシーを直接作成
    from stable_baselines3.common.policies import ActorCriticPolicy
    
    # 常に一定の学習率を返すダミー関数
    def constant_lr(_):
        return 0.001

    custom_policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=constant_lr,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        batch_size=64,
        policy=custom_policy,
        rng=np.random.default_rng(42) # 乱数シードを固定
    )

    print("--- BC Training Start (Net: 256x256, Epochs: 1000) ---")
    bc_trainer.train(n_epochs=300)
    
    os.makedirs("models", exist_ok=True)
    torch.save(bc_trainer.policy, "models/bc_policy.pt")
    print("Policy saved to models/bc_policy.pt")

if __name__ == "__main__":
    train()
