import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class RobotHandEnv(gym.Env):
    def __init__(self, hardware_interface):
        super(RobotHandEnv, self).__init__()
        self.hw = hardware_interface

        # Action: [角度5, CNC Z 1] = 6次元
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Observation: [MMS1 3, MMS2 3, MMS3 3, CNC Z 1, Radius 1, Angles 5] = 16次元
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # 範囲設定 (Negative=Down 座標系)
        self.cnc_min_z = -32.0
        self.cnc_max_z = 0.0
        self.home_z = -15.0
        
        # 試行の設定
        self.max_steps = 200 
        self.current_step = 0
        
        # 内部状態
        self.start_xy = [0.0, 0.0]
        self.last_sent_z = None
        self.smoothed_action = None
        self.alpha = 0.3 # 小さいほど滑らか (0.1~0.3推奨)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.smoothed_action = np.zeros(6)
        
        # 統計リセット
        self.episode_forces = []
        self.episode_penalties = 0.0
        
        # 現在のX,Y位置を取得して保持
        pos, _ = self.hw.cnc.get_current_pos()
        self.start_xy = [pos[0], pos[1]]
        self.last_sent_z = self.home_z

        # ホーム(Z=-15mm)に戻る
        print("\n--- Resetting to Home Position ---")
        self.hw.move_hand([90]*5)
        self.hw.cnc.move_to([self.start_xy[0], self.start_xy[1], self.home_z])
        self.hw.wait_cnc()
        
        # 次の試行のための準備待ち
        input("ボールをセットし、準備ができたら Enter キーを押して次の試行を開始してください...")
        
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
        
        # 2. 値の分解
        hand_angles = ((self.smoothed_action[:5] + 1.0) / 2.0 * 160.0 + 10.0).astype(int)
        target_z = (self.smoothed_action[5] + 1.0) / 2.0 * (self.cnc_max_z - self.cnc_min_z) + self.cnc_min_z
        
        # 3. ハードウェアへの指令
        self.hw.move_hand(hand_angles)
        
        # CNCの指示を間引く
        if self.last_sent_z is None or abs(target_z - self.last_sent_z) > 0.5:
            self.hw.cnc.move_to([self.start_xy[0], self.start_xy[1], target_z])
            self.last_sent_z = target_z

        # 4. 観測値の生成
        self.hw.update_sensor_data(update_cnc=False)
        self.hw._latest_cnc_pos[2] = target_z 
        obs = self.hw.get_observation()

        # 5. 報酬計算と終了判定
        done = self.current_step >= self.max_steps

        # 接触報酬
        contact_count = sum(self.hw.contact_statuses)
        reward = contact_count * 0.1 

        # 荷重ペナルティ (不感帯なしの直接ペナルティに戻す)
        total_force = sum([abs(fz) for fz in self.hw.last_fz_values])
        force_penalty = total_force * 0.02 
        reward -= force_penalty
        
        # 統計更新
        self.episode_forces.append(total_force)
        self.episode_penalties += force_penalty

        success = False
        if done:
            self.hw.wait_cnc()
            print("\n>>> Time up! Attempting Auto-Lift...")
            success = self.hw.auto_lift(15.0)
            
            # エピソード統計の表示
            avg_f = np.mean(self.episode_forces) if self.episode_forces else 0.0
            max_f = np.max(self.episode_forces) if self.episode_forces else 0.0
            print(f"\n--- Episode Summary ---")
            print(f" Success: {success}")
            print(f" Avg Force: {avg_f:.2f} N")
            print(f" Max Force: {max_f:.2f} N")
            print(f" Total Force Penalty: -{self.episode_penalties:.2f}")
            print(f"-----------------------\n")
            
            if success:
                reward += 100.0
        
        info = {
            "success": success, 
            "step": self.current_step,
            "total_force": total_force,
            "forces": list(self.hw.last_fz_values)
        }

        return obs, reward, done, False, info


    def close(self):
        self.hw.disconnect()
