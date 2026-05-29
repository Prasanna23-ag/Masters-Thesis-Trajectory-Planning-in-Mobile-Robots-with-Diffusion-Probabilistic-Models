import gym
from gym import spaces
import numpy as np

class CustomTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = np.array([100.0, 100.0], dtype=np.float32)
        self._target = self.goal.copy()  # Required for maze2d_set_terminals
        self._max_episode_steps = 1000  # Required for D4RL-style logic
        self.max_episode_steps = 1000   # Optional but good for consistency

    def reset(self):
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        self.state = self.state + action
        done = np.linalg.norm(self.state - self.goal) < 1.0
        reward = -np.linalg.norm(action)
        if done:
            reward += 10
        return self.state, reward, done, {}
