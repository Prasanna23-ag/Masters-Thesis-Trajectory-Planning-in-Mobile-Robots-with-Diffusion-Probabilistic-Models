import os
import numpy as np
import pandas as pd
from gym import utils
import gym

class CustomTestEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        # Load path from assets
        asset_path = os.path.join(os.path.dirname(__file__), 'assets/path.csv')
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"Path file not found: {asset_path}")
        self.path = pd.read_csv(asset_path)[['x', 'y']].values
        self.goal = self.path[-1]
        self.init_pos = self.path[0]
        self.max_steps = 3000

        # Define action and observation space
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(15,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Internal state
        self.state = None
        self.step_count = 0

        # Maze2D compatibility
        self._target = self.goal.copy()
        self._max_episode_steps = self.max_steps
        self.max_episode_steps = self.max_steps

        # ✅ Updated for full map rendering
        self.maze_arr = np.zeros((100, 100), dtype=np.int32)

        # Draw the "L" shaped obstacle
        self.maze_arr[20:80, 20:30] = 1  # vertical bar of L
        self.maze_arr[70:80, 20:70] = 1  # horizontal bar of L

        # Mark the goal cell
        goal_x, goal_y = np.clip(self.goal.astype(int), 0, 99)
        self.maze_arr[goal_x, goal_y] = 10

        utils.EzPickle.__init__(self)

    def reset(self):
        self.state = np.concatenate([
            self.init_pos.copy(),               # root x, y
            np.array([0.5]),                    # root z
            np.zeros(12, dtype=np.float32)      # joint angles and velocities
        ])
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        root_pos = self.state[:2]
        new_pos = np.clip(root_pos + action[:2], 0, 100)
        self.state[:2] = new_pos
        self.state[3:] = np.clip(action, -1, 1)

        forward_reward = np.linalg.norm(new_pos - root_pos)
        ctrl_cost = 0.5 * np.square(action).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward

        done = False
        if np.linalg.norm(new_pos - self.goal) < 1.0:
            reward += 10
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        self.step_count += 1
        return self._get_obs(), reward, done, {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_survive': survive_reward
        }

    def _get_obs(self):
        return self.state.copy()

    def viewer_setup(self):
        pass

    def get_dataset(self, load_path=None):
        import h5py
        path = load_path or os.path.join(os.path.dirname(__file__), 'assets/d4rl_path_dataset.hdf5')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with h5py.File(path, 'r') as f:
            observations = f['observations'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]
            timeouts = f['timeouts'][:] if 'timeouts' in f else np.zeros_like(terminals)

            # Ensure next_observations is aligned and safe
            if 'next_observations' in f:
                next_observations = f['next_observations'][:]
            else:
                if len(observations) > 1:
                    next_observations = observations[1:]
                    observations = observations[:-1]
                    actions = actions[:-1]
                    rewards = rewards[:-1]
                    terminals = terminals[:-1]
                else:
                    next_observations = observations.copy()

            dataset = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'terminals': terminals,
                'timeouts': timeouts,
                'next_observations': next_observations
            }
        return dataset
