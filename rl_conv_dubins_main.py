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
import meta_learner_env

# %%
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)
importlib.reload(gpt_class_exactgpmodel)
importlib.reload(meta_learner_env)
# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
# Device selection supporting CUDA, MPS (Apple Silicon), or CPU
#if torch.backends.mps.is_available():
#    self.device = torch.device("mps")
device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

turn_radius = 25
#channels = np.array([0, 1, 0, 0, 0]) # only explore
channels = np.array([1, 0, 0, 0, 0]) # only gas

env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[10.0, 10.0, 1.0], channels=channels, turn_radius=turn_radius, timer=False, debug=False, device=device)

buffer_size = 400_000                      # how many transitions

replay_buffer = rl_gas_survey_dubins_env.CpuDictReplayBuffer(
    buffer_size       = buffer_size,
    observation_space = env.observation_space,
    action_space      = env.action_space,
    device            = "cpu",           # storage
    sample_device     = device,          # default target device
    optimize_memory_usage = False
)

# %%
host = socket.gethostname().split('.')[0]
if host in ['dunder', 'cupid', 'dancer', 'rudolph', 'dasher']:
    parent_dir = "/projects/robin/users/ivarkriw/in5490"
    log_interval = 100
else:
    parent_dir = '.'
    log_interval = 30

models_parent = parent_dir + "/models"
logs_parent = parent_dir + "/logs"

current_dir = f"/{int(time.time())}_{host}"
models_dir = models_parent + current_dir
logs_dir = logs_parent + current_dir

load = False
if load:
    load_time = '1749667471_dunder' # timestamp_host
    load_model = '0_1323' # zip-file without extension
    save_prefix = '1_'
    models_dir = f"{models_parent}/{load_time}"

    agent = DQN.load(f"{models_dir}/{load_model}", env=env, device=env.device)
    try:
        agent.load_replay_buffer(f"{models_dir}/buffer.pkl")
    except:
        print(f'Could not load replay buffer from {models_dir}/buffer.pkl')

    print(f'Loaded model from {models_dir}/{load_model}')
else:
    save_prefix = '0_'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

#    policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=256))
    policy_kwargs = dict(
        features_extractor_class=rl_gas_survey_dubins_env.MapPlusLocExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    # Load model from zip-file
    # loading saved models
    path = models_parent + "/"
    explorer = path + "explorer_model" + "/0_1019953"
    exploiter = path + "exploiter_model" + "/0_3340000"
    agent1 = DQN.load(explorer, env=env, device=env.device)
    agent2 = DQN.load(exploiter, env=env, device=env.device)
    agent = DQN(
        "MultiInputPolicy",
        env=meta_learner_env.MetaSelectEnv(env, agent1, agent2),  # env returns {"map": ..., "loc": ...}
        device=env.device,
        buffer_size=buffer_size,
        batch_size=256,
        learning_rate=3e-4,
        learning_starts=256,
        tau=0.005,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logs_dir
    )

    agent.replay_buffer = replay_buffer          # overwrite in place

# %%
#TIMESTEPS = 5_000
TIMESTEPS = 10_000

while True:
    agent.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        log_interval=log_interval,
        tb_log_name=f'DQN'
        )
    
    agent.save(f"{models_dir}/{save_prefix}{env.total_steps}")
    agent.save_replay_buffer(f"{models_dir}/buffer.pkl")

# %%
# Run an episode (with a random agent)
agent.exploration_rate = 0.9

obs, _ = env.reset()
done = False
rewards = np.array([])
q_values = []
while not done:
    if env.debug:
        #q_vec = rl_gas_survey_discrete_env.get_q_values(agent, obs)
        q_vec = rl_gas_survey_dubins_env.get_q_values(agent, obs)
        q_values.append(q_vec)

    action, _step = agent.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards = np.append(rewards, reward)
    done = terminated or truncated

q_values = np.vstack(q_values)

# %%
# Plot rewards and results from one episode
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.obs_truth, path=env.sampled_coords[:env.sample_idx], value_title='obs_truth')
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_var_norm, path=env.sampled_coords[:env.sample_idx], value_title='pred_var')
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_mu_norm, path=env.sampled_coords[:env.sample_idx], value_title='pred_mu')

q_act = ['left', 'straight', 'right']
fig, ax = plt.subplots(figsize=(4.5, 2.2), dpi=300)   # fits two-column journals
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
