# env_balance_guided.py

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class BalanceEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("S-1-Roller_model.xml")
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode

        self.dt = self.model.opt.timestep
        self.time = 0.0

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")

        self.observation_space = spaces.Box(
            low=np.array([math.radians(-30), math.radians(-250), -300.0], dtype=np.float32),
            high=np.array([math.radians(30), math.radians(250), 300.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_hinge")
        self.qvel_index = self.model.jnt_dofadr[self.joint_id]
        self.np_random = np.random.default_rng()

    def quat_to_pitch(self, q):
        w, x, y, z = q
        sinp = 2.0 * (w * y - z * x)
        cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(sinp, cosp)

    def _get_obs(self):
        quat = self.data.qpos[3:7]
        theta_y = -self.quat_to_pitch(quat)
        omega_y = -self.data.qvel[4]
        wheel_speed = self.data.qvel[self.qvel_index]
        return np.array([theta_y, omega_y, wheel_speed], dtype=np.float32)

    def step(self, action):
        # --- time更新 ---
        self.time += self.dt

        # PPOの出力（[-1, 1]）をスケールしてMuJoCoに渡す
        scaled_action = np.clip(action[0], -1.0, 1.0)
        self.data.ctrl[0] = float(scaled_action)

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        theta, omega, wheel_speed = obs

        theta_max = math.radians(30)
        omega_max = math.radians(250)
        wheel_max = 300.0

        theta_n = theta / theta_max
        omega_n = omega / omega_max
        wheel_n = wheel_speed / wheel_max

        reward = (
            - theta_n**2
            - omega_n**2
            - wheel_n**2
        )

        # --- ログ（100ステップに1回だけ表示） ---
        self.timestep_count += 1 
        if self.timestep_count % 100 == 0:
            print(f"[STEP] time={self.time:.3f}, pitch={np.degrees(theta):>6.2f}, reward={reward:6.3f}")

        terminated = bool(abs(theta) > np.pi / 6)  # 30度以上で終了
        truncated = False

        return obs, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.time = 0.0

        # 完全ゼロ初期化
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.timestep_count = 0

        # ランダムなY軸傾き
        theta = self.np_random.uniform(low=math.radians(-3.0), high=math.radians(3.0))
        half_theta = theta / 2
        self.data.qpos[3] = np.cos(half_theta)  # w
        self.data.qpos[5] = np.sin(half_theta)  # y
 
        # Y軸の角速度
        omega_y = self.np_random.uniform(low=-0.1, high=0.1)
        self.data.qvel[4] = omega_y 

        # ホイール角速度
        wheel_speedIni = self.np_random.uniform(low=-0.05, high=0.05)
        self.data.qvel[self.qvel_index] = wheel_speedIni


        # 物理整合性を更新
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        return obs, {}


    def render(self):
        # 学習・実行ともにここでは何もしない
        return

    def close(self):
        # viewer をここで管理していないので何もしない
        return
