import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class RobotHandEnv(gym.Env):
    def __init__(self, hardware_interface, reach_model=None):
        super(RobotHandEnv, self).__init__()
        self.hw = hardware_interface
        self.reach_model = reach_model # 半径 -> 到達高度 を予測するモデル

        # Action: [角度5] = 5次元 (Zは動かさない)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        # Observation: [MMS Fz*3, Radius, Angles*5] = 9次元
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # 範囲設定
        self.cnc_min_z = -32.0
        self.cnc_max_z = 0.0
        self.target_z = 0.0 # REACHモデルで決定される目的地
        
        # 試行の設定
        self.max_steps = 500 
        self.current_step = 0
        
        # 内部状態
        self.start_xy = [0.0, 0.0]
        self.smoothed_action = None
        self.alpha = 1# 1.0 に戻してAIの出力をダイレクトに反映させる

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.smoothed_action = np.zeros(5)
        
        # 統計リセット
        self.episode_forces = []
        self.episode_penalties = 0.0
        
        # 現在のX,Y位置を取得して保持
        pos, _ = self.hw.cnc.get_current_pos()
        self.start_xy = [pos[0], pos[1]]

        # 1. REACHフェーズ: 目標高度の決定
        if self.reach_model is not None:
            import torch
            # 半径を正規化(10mm->1.0)して入力
            radius_tensor = torch.as_tensor([[self.hw.ball_radius / 10.0]], dtype=torch.float32)
            with torch.no_grad():
                pred_z_norm = self.reach_model(radius_tensor).item()
                # Zの復元: z = (norm_z - 1) * 16
                self.target_z = (pred_z_norm - 1.0) * 16.0
        else:
            # モデルがない場合のデフォルト (安全な高さ)
            self.target_z = -25.0

        print(f"\n--- REACH Phase: Moving to Target Z ({self.target_z:.1f}mm) ---")
        self.hw.move_hand([90]*5)
        self.hw.cnc.move_to([self.start_xy[0], self.start_xy[1], self.target_z])
        self.hw.wait_cnc()
        
        # 次の試行のための準備待ち
        input("ボールの位置を確認し、準備ができたら Enter を押して把持(GRASP)を開始...")
        
        # 初期位置到達後にセンサーをゼロ点調整 (Tare)
        self.hw.tare_sensors()
        self.hw.update_sensor_data(update_cnc=True)
        return self.hw.get_observation(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. AIのアクションをスムージング (EMA)
        if self.smoothed_action is not None:
            self.smoothed_action = self.alpha * action + (1 - self.alpha) * self.smoothed_action
        else:
            self.smoothed_action = action
        
        # 2. 値の分解 (AIの出力: +1.0 = 閉じ, -1.0 = 開き に統一)
        hand_angles = np.zeros(5, dtype=int)
        hand_angles[0] = int(self.smoothed_action[0] * 75.0 + 95.0) # 親指
        hand_angles[1:] = (100.0 - (self.smoothed_action[1:5] * 80.0)).astype(int) # 他
        hand_angles = np.clip(hand_angles, 10, 175)
        
        # 3. ハードウェアへの指令 (Zは固定)
        self.hw.move_hand(hand_angles)
        
        # 4. 観測値の生成
        self.hw.update_sensor_data(update_cnc=False)
        obs = self.hw.get_observation()

        # 報酬計算と終了判定
        done = self.current_step >= self.max_steps

        reward = 0.0

        # 荷重ペナルティ (強すぎる力を厳しく抑制)
        total_force = sum([abs(fz) for fz in self.hw.last_fz_values])
        force_penalty = total_force * 0.4 
        reward -= force_penalty

        self.episode_forces.append(total_force)


        success = False
        if done:
            print("\n>>> Time up! Attempting Auto-Lift...")
            success = self.hw.auto_lift(15.0)
            
            print(f"\n--- Episode Summary ---")
            print(f" Success: {success}")
            print(f"-----------------------\n")
            if success:
                reward += 50.0
        
        info = {
            "success": success, 
            "step": self.current_step,
            "total_force": total_force,
            "forces": list(self.hw.last_fz_values)
        }

        return obs, reward, done, False, info

    def close(self):
        self.hw.disconnect()
