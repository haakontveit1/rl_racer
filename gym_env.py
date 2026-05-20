"""Gymnasium-compatible wrapper around RacerEnv.

Translates between RacerEnv's tuple-based API and Gymnasium's expectations:
- numpy observations
- 5-tuple step return (obs, reward, terminated, truncated, info)
- 2-tuple reset return (obs, info)
- declared observation_space and action_space

Action space is continuous Box(2,): [steer, throttle].
- steer in [-1, 1]: -1 = full left, +1 = full right
- throttle in [0, 1]: 0 = coast, 1 = full gas
"""

import gymnasium as gym
import numpy as np

from environment.env import RacerEnv
from environment.action import Action


class GymRacer(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, waypoints=None, randomize_start=False):
        super().__init__()
        self.env = RacerEnv(waypoints=waypoints, randomize_start=randomize_start)

        # Obs layout (11 dims):
        #   [0..4]  5 rays in [0, 1]
        #   [5]     forward speed   in [-2, 2]
        #   [6]     lateral speed   in [-2, 2]
        #   [7]     last_steer      in [-1, 1]
        #   [8]     last_throttle   in [0, 1]
        #   [9]     angular_velocity (normalized) in [-2, 2]
        #   [10]    signed lateral offset from centerline (normalized) in [-3, 3]
        obs_low = np.array(
            [0.0] * 5 + [-2.0, -2.0, -1.0, 0.0, -2.0, -3.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0] * 5 + [2.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # [steer, throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        obs, reward, done, info = self.env.step(Action(steer=steer, throttle=throttle))
        terminated = bool(info["crossed_finish"])
        truncated = bool(info["timed_out"])
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info
