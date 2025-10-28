# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np
import gpytorch
import pickle


import importlib
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


import time
import matplotlib.pyplot as plt
import seaborn as sns

import rl_scenario_bank
import rl_gas_survey_dubins_env
import chem_utils
import gpt_class_exactgpmodel
import meta_learner_env

# Load scenarios
bank = rl_scenario_bank.ScenarioBank(data_dir='.')
envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
# Setup environment
env_device = torch.device("cpu")
action_mode = ['relative', 20, 20]
channels = np.array([1, 0, 0, 0, 0])
turn_radius = 25


env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[10.0, 10.0, 1.0], channels=channels, turn_radius = turn_radius, timer=False, debug=True, device=env_device)


agent_explore = DQN.load("exploration_last", env=env, device=env.device,load_replay_buffer=False)
agent_exploit = DQN.load("0_3350000", env=env, device=env.device,load_replay_buffer=False)
meta_env = meta_learner_env.MetaSelectEnv(env, agent_explore, agent_exploit, chunk_len=5)

#agent_meta = DQN.load("meta", env=meta_env, device=env.device,load_replay_buffer=False)

agent_meta = DQN.load("meta", env=meta_env,
    custom_objects={
        "replay_buffer": None,           # don't create the old huge buffer
        "buffer_size": 10_000,          # override the size in case it's used in setup
    })



# %%
# Run an episode
obs, _ = meta_env.reset()
done = False
rewards = np.array([])
q_values = []

step_num = 0

#while not done:
steps = 50 
for _ in range(steps//meta_env.chunk_len):


    agent_decision = agent_meta.predict(obs, deterministic=True)
    q_vec = rl_gas_survey_dubins_env.get_q_values(agent_meta, obs)
    q_values.append(q_vec)


    start = time.time()
    obs, reward, terminated, truncated, info = meta_env.step(agent_decision[0])
    rewards = np.append(rewards, reward)
    done = terminated or truncated

    end = time.time()
    print(f"Step time: {end-start:.3f} s")
    step_num += 1

print("Done")

q_values = np.vstack(q_values)

# %%
# plotting
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_var_norm, path=env.sampled_coords[:env.sample_idx])
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_mu_norm_clipped, path=env.sampled_coords[:env.sample_idx])
env.plot_env(path=env.sampled_coords[:env.sample_idx])

q_act = ['left', 'straight', 'right']
fig, ax = plt.subplots(figsize=(4.5, 2.2), dpi=300) # fits two-column journals
steps = np.arange(len(rewards))
ax.plot(steps, rewards, label="reward", linewidth=0.6)
for i in range(q_values.shape[1]):
    ax.plot(steps, q_values[:, i], label=f"{q_act[i]}", linewidth=0.6)

ax.set_xlabel("Step", fontsize=8)
ax.set_ylabel("Reward", fontsize=8)
#ax.set_ylim(-1, 1)
ax.tick_params(axis="both", labelsize=7)
ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=6, ncol=q_values.shape[1]+1)
fig.tight_layout()

# %%
