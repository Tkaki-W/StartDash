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
        # Observation: [MMS Fx,Fy,Fz 3, CNC Z 1, Radius 1, Angles 5]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # 範囲設定 (Negative=Down 座標系)
        self.cnc_min_z = -35.0
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
        
        # 現在のX,Y位置を取得して保持
        pos, _ = self.hw.cnc.get_current_pos()
        self.start_xy = [pos[0], pos[1]]
        self.last_sent_z = self.home_z

        # ホーム(Z=-15mm)に戻る
        self.hw.move_hand([90]*5)
        self.hw.cnc.move_to([self.start_xy[0], self.start_xy[1], self.home_z])
        self.hw.wait_cnc()
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
        
        # CNCの指示を間引く (0.2mm以上の変化時のみ)
        if self.last_sent_z is None or abs(target_z - self.last_sent_z) > 1:
            self.hw.cnc.move_to([self.start_xy[0], self.start_xy[1], target_z])
            self.last_sent_z = target_z
            time.sleep(0.01) # 通信の安定化
        
        # 4. 観測値の生成
        self.hw.update_sensor_data(update_cnc=False)
        self.hw._latest_cnc_pos[2] = target_z 
        obs = self.hw.get_observation()

        # 5. 終了判定
        done = self.current_step >= self.max_steps
        success = False
        reward = 0.0

        if done:
            self.hw.wait_cnc()
            print("\n>>> Time up! Attempting Auto-Lift...")
            success = self.hw.auto_lift(15.0)
            reward = 100.0 if success else 0.0
        
        return obs, reward, done, False, {"success": success, "step": self.current_step}

    def close(self):
        self.hw.disconnect()
