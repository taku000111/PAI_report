# -*- coding: utf-8 -*-

import os
import multiprocessing as mp

# ==== 環境変数（OpenBLAS / MKL / MuJoCo / PyTorch のスレッド数制御） ====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")  # Windows なら glfw が無難

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from env_balance import BalanceEnv  # イッチの env_balance.py 前提


def make_env(rank: int):
    def _init():
        return BalanceEnv()
    return _init


if __name__ == "__main__":
 
    mp.set_start_method("spawn", force=True)

    # ==== CPU コア数から並列環境数を決める ====
    logical = os.cpu_count() or 4          # 例: 12（6コア12スレッド）
    phys_est = max(1, logical // 2)        # 例: 6（物理コアの推定）
    # 倒立振子は軽いので 2〜6 環境に制限
    N_ENVS = max(2, min(phys_est, 6))

    print(f"[INFO] logical_cpu={logical}, phys_est={phys_est}, N_ENVS={N_ENVS}")

    # PyTorch のスレッド数を絞る（多いと逆に遅くなる）
    torch.set_num_threads(max(1, phys_est // 2))

    # ==== VecEnv 構築 ====
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # ==== PPO モデル定義 ====
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": [64, 64],
            "log_std_init": 0.3,
        },
        learning_rate=3e-4,
        n_steps=1024,         
        batch_size=512,     
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        verbose=1,
    )

    # ==== 学習実行 ====
    TOTAL_STEPS = 1_000_000
    print(f"[INFO] Start learning: total_timesteps={TOTAL_STEPS}")
    model.learn(total_timesteps=TOTAL_STEPS)

    # ==== 保存 ====
    save_path = "ppo_balance"
    model.save(save_path)
    print(f"[INFO] Saved model to: {save_path}.zip")

    env.close()
    print("[INFO] Training finished.")
