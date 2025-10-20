# %%
import random
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np
import gpytorch

import importlib
import torch
from stable_baselines3 import DQN

import time
import matplotlib.pyplot as plt
import seaborn as sns

import rl_scenario_bank
import rl_gas_survey_dubins_env
import chem_utils
import gpt_class_exactgpmodel

# %%
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)
importlib.reload(gpt_class_exactgpmodel)

# %%
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
channels = np.array([0, 1, 0, 0, 0])
turn_radius = 25
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0, 1.0], channels=channels, turn_radius = turn_radius, timer=False, debug=True, device=env_device)

# %%
# Load model from zip-file
# loading saved models
path = 'models/'
explorer = path + "0_1019953"
exploiter = path + "0_3330000"
agent1 = DQN.load(explorer, env=env, device=env.device)
agent2 = DQN.load(exploiter, env=env, device=env.device)

# %%
# Run an episode
class random_meta_agent:
    def __init__(self, explorer, exploiter):
        self.explorer = explorer
        self.exploiter = exploiter
        self.model = None

    def choose_model(self):
        random_choice = random.randint(0, 1)
        if random_choice == 1:
            return self.exploiter
        else:
            return self.explorer
        
    def predict(self, obs):
        if self.model is None:
            self.model = self.choose_model()
        action, _step = self.model.predict(obs, deterministic=True)
        return action, _step
    
random_agent = random_meta_agent(agent1, agent2)    
obs, _ = env.reset()
q_values = []
rewards = np.array([])  

for i in range(100):
    if i%10 == 0:
        random_agent.model = random_agent.choose_model()
    
    if env.debug:
        q_vec = rl_gas_survey_dubins_env.get_q_values(random_agent.model, obs)
    q_values.append(q_vec)

    action, _step = random_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(int(action))
    rewards = np.append(rewards, reward)
    done = terminated or truncated

    print(f"Episode {i+1}/100, model: {random_agent.model}, reward: {np.sum(rewards):.2f}, steps: {len(rewards)}")
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
