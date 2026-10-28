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
            low=np.array([math.radians(-90), math.radians(-250), -300.0], dtype=np.float32),
            high=np.array([math.radians(90), math.radians(250), 300.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_hinge")
        self.qvel_index = self.model.jnt_dofadr[self.joint_id]

        # reset()ごとのドメインランダマイゼーション用に基準値を保持する。
        self.motor_id = 0
        self.base_motor_gear = self.model.actuator_gear[self.motor_id].copy()
        self.base_body_mass = self.model.body_mass.copy()
        self.base_body_inertia = self.model.body_inertia.copy()
        self.base_wheel_damping = float(self.model.dof_damping[self.qvel_index])

        # 実機でKp=3が必要だったことを踏まえ、角度観測倍率を1～3で変える。
        self.angle_obs_scale = 1.0
        self.angle_bias = 0.0
        self.action_delay_steps = 0
        self.action_history = [0.0, 0.0]

    def quat_to_pitch(self, q):
        w, x, y, z = q
        sinp = 2.0 * (w * y - z * x)
        cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(sinp, cosp)

    def _get_obs(self):
        theta_y, omega_y, wheel_speed = self._get_true_state()

        # 方策に渡すセンサ相当の値に、固定バイアスと小さな測定ノイズを加える。
        theta_y = (
            (theta_y + self.angle_bias) * self.angle_obs_scale
            + self.np_random.normal(0.0, math.radians(0.15))
        )
        omega_y += self.np_random.normal(0.0, math.radians(0.5))
        wheel_speed += self.np_random.normal(0.0, 0.5)
        return np.array([theta_y, omega_y, wheel_speed], dtype=np.float32)

    def _get_true_state(self):
        quat = self.data.qpos[3:7]
        theta_y = -self.quat_to_pitch(quat)
        omega_y = -self.data.qvel[4]
        wheel_speed = self.data.qvel[self.qvel_index]
        return theta_y, omega_y, wheel_speed

    def step(self, action):
        # PPOの出力を[-1, 1]に制限し、エピソードごとの遅延を模擬する。
        requested_action = float(np.clip(action[0], -1.0, 1.0))
        self.action_history.insert(0, requested_action)
        applied_action = self.action_history[self.action_delay_steps]
        self.action_history = self.action_history[:2]
        self.data.ctrl[0] = applied_action

        mujoco.mj_step(self.model, self.data)
        self.time += self.dt

        # 報酬・転倒判定にはノイズのない物理状態を使う。
        theta, omega, wheel_speed = self._get_true_state()
        obs = self._get_obs()

        # 直立角度を優先しつつ、角速度・車輪速度・入力の過大化も抑える。
        theta_ref = math.radians(5.0)
        omega_ref = math.radians(50.0)
        wheel_ref = 100.0
        cost = (
            2.0 * (theta / theta_ref) ** 2
            + 0.2 * (omega / omega_ref) ** 2
            + 0.05 * (wheel_speed / wheel_ref) ** 2
        )
        reward = float(np.exp(-cost) - 0.005 * applied_action**2)

        # --- ログ（100ステップに1回だけ表示） ---
        self.timestep_count += 1 
        if self.timestep_count % 100 == 0:
            print(f"[STEP] time={self.time:.3f}, pitch={np.degrees(theta):>6.2f}, reward={reward:6.3f}")

        terminated = bool(abs(theta) > math.radians(20.0))
        truncated = bool(self.time >= 10.0)
        if terminated:
            reward = -5.0

        return obs, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 毎エピソード、必ず基準値から物理パラメータを再設定する。
        motor_scale = self.np_random.uniform(0.8, 1.2)
        mass_scale = self.np_random.uniform(0.9, 1.1)
        damping_scale = self.np_random.uniform(0.5, 1.5)
        self.model.actuator_gear[self.motor_id] = self.base_motor_gear * motor_scale
        self.model.body_mass[:] = self.base_body_mass
        self.model.body_inertia[:] = self.base_body_inertia
        self.model.body_mass[1:] *= mass_scale
        self.model.body_inertia[1:] *= mass_scale
        self.model.dof_damping[self.qvel_index] = (
            self.base_wheel_damping * damping_scale
        )
        mujoco.mj_setConst(self.model, self.data)

        # Kp=1～3相当の角度入力倍率、ゼロ点ずれ、0～8msの行動遅延。
        self.angle_obs_scale = self.np_random.uniform(1.0, 3.0)
        self.angle_bias = self.np_random.uniform(
            -math.radians(0.5), math.radians(0.5)
        )
        self.action_delay_steps = int(self.np_random.integers(0, 3))
        self.action_history = [0.0, 0.0]

        mujoco.mj_resetData(self.model, self.data)

        self.time = 0.0

        # 完全ゼロ初期化
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.timestep_count = 0

        # ランダムなY軸傾き
        theta = self.np_random.uniform(low=math.radians(-5.0), high=math.radians(5.0))
        half_theta = theta / 2
        self.data.qpos[3] = np.cos(half_theta)  # w
        self.data.qpos[5] = np.sin(half_theta)  # y
 
        # Y軸の角速度
        omega_y = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.qvel[4] = omega_y 

        # ホイール角速度
        wheel_speedIni = self.np_random.uniform(low=-0.1, high=0.1)
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
