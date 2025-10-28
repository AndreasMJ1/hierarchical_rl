from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium import spaces


# Meta-selection wrapper env: action=0 selects model_a (explorer), action=1 selects model_b (exploiter)
class MetaSelectEnv(gym.Env):
    def __init__(self, base_env: gym.Env, model_a: DQN, model_b: DQN, chunk_len: int = 5):
        super().__init__()
        self.base_env = base_env
        self.model_a = model_a  # explorer → channel 1
        self.model_b = model_b  # exploiter → channel 0
        self.chunk_len = int(chunk_len)

        # Observation & action spaces
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Discrete(2)
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def _get_child_obs(self, obs, select_action: int):
        """
        Returns the single-channel observation for the selected model.
        Model A (explorer) uses channel 1.
        Model B (exploiter) uses channel 0.
        """
        if int(select_action) == 0:
            # Explorer (model_a) → channel 1
            channel_idx = 1
        else:
            # Exploiter (model_b) → channel 0
            channel_idx = 0

        # Copy the observation dict and replace only the map slice
        child_obs = dict(obs)  # shallow copy preserves other keys like "hdg", "vel", etc.
        child_obs["map"] = obs["map"][channel_idx:channel_idx + 1, :, :]

        return child_obs


    def step(self, select_action: int):
        # Select the model based on meta-action
        if int(select_action) == 0:
            chosen_model = self.model_a  # explorer
            model_label = "explorer"
        else:
            chosen_model = self.model_b  # exploiter
            model_label = "exploiter"

        total_reward = 0.0
        terminated = False
        truncated = False
        low_actions = []
        info_last = {}
        obs = self._last_obs

        # Execute a sequence (chunk) of low-level actions
        for _ in range(self.chunk_len):
            # Slice correct channel for the selected child model
            child_obs = self._get_child_obs(obs, select_action)
            low_action, _ = chosen_model.predict(child_obs, deterministic=True)
            low_actions.append(int(low_action))

            # Step base environment
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
        info_out["selected_model"] = model_label
        info_out["low_actions"] = low_actions
        info_out["chunk_len"] = self.chunk_len

        return obs, float(total_reward), terminated, truncated, info_out
