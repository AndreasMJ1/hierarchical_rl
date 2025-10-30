# %%
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
#env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0, 1.0], channels=channels, turn_radius = turn_radius, timer=False, debug=False, device=env_device)


env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[10.0, 10.0, 1.0], channels=channels, turn_radius = turn_radius, timer=False, debug=False, device=env_device)

# %%
# Load model from zip-file
load_model = '1758560582_rudolph_0_3759974'
models_dir = f"models"
model_load = "exploit"
agent = DQN.load(f"{model_load}", env=env, device=env.device)

# %%
# Run an episode
done = False
rewards = np.array([])

coverage = []
max_environments = 100

mu_mapping = True 
mu = []
for environment in range(max_environments):
    obs, _ = env.reset()

    for _ in range(100):        
        action, _step = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards = np.append(rewards, reward)

    if mu_mapping:
        mask = env.pred_mu < env.min_concentration
        env.pred_mu[mask] = env.min_concentration
        mu.append(np.mean(abs(env.pred_mu-env.obs_truth)))

    #After every environment
    total_gas = (((env.values - env.min_concentration) / (env.max_concentration - env.min_concentration) * 255) >= 5).sum()

    total_gas_smp = (((env.sampled_vals[:env.sample_idx] - env.min_concentration) / (env.max_concentration - env.min_concentration) * 255) >= 5).sum()
    percentage_gas_sampled = ((total_gas_smp) / (total_gas)) * 100
    coverage.append(percentage_gas_sampled)

    if mu_mapping:
        print(f"Mean absolute error in GP mean prediction: {mu[-1]:.4f}")
        np.save("mu_dqn_exploit.npy", np.array(mu))
    np.save("coverage_dqn_exploit.npy", np.array(coverage))


    if percentage_gas_sampled >= max(coverage):
        env.plot_env(path=env.sampled_coords[:env.sample_idx])

    print(f"Percentage of total gas sampled: {percentage_gas_sampled:.2f}%")

np.save("coverage_dqn_exploit.npy", np.array(coverage))



# %%
