# %%
from stable_baselines3 import DQN

import gymnasium as gym
from gymnasium import spaces


# %%
# Meta-selection wrapper env: action=0 selects agent1 (explorer), action=1 selects agent2 (exploiter)
class MetaSelectEnv(gym.Env):
    def __init__(self, base_env: gym.Env, model_a: DQN, model_b: DQN, chunk_len: int = 5):
        super().__init__()
        self.base_env = base_env
        self.model_a = model_a
        self.model_b = model_b
        self.chunk_len = int(chunk_len)
        # Observation space identical to base env, actions are binary selection
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Discrete(2)
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, select_action: int):
        # Choose which low-level model to use for this chunk
        chosen = self.model_b if int(select_action) == 1 else self.model_a

        total_reward = 0.0
        terminated = False
        truncated = False
        low_actions = []
        info_last = {}
        obs = self._last_obs
        print(f"MetaSelectEnv: selected model {'B' if int(select_action) == 1 else 'A'} for next {self.chunk_len} steps")

        for _ in range(self.chunk_len):
            # Decide low-level action using the chosen model
            low_action, _ = chosen.predict(obs, deterministic=True)
            low_actions.append(int(low_action))
            # Step the base env
            obs, reward, term, trunc, info = self.base_env.step(int(low_action))
            total_reward += float(reward)
            info_last = info
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        self._last_obs = obs
        # Aggregate info
        info_out = dict(info_last)
        info_out["selected_model"] = "B" if int(select_action) == 1 else "A"
        info_out["low_actions"] = low_actions
        info_out["chunk_len"] = self.chunk_len

        return obs, float(total_reward), terminated, truncated, info_out


