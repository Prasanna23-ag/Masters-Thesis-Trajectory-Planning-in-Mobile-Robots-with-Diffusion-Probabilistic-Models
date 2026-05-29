import os
import numpy as np
import h5py
from gym import utils
import matplotlib.pyplot as plt
import gym

class TurtleBotEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        # Load dataset from assets
        asset_path = os.path.join(os.path.dirname(__file__), 'assets/birrt-dataset-v0.hdf5')
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"Dataset file not found: {asset_path}")
        self.asset_path = asset_path
        self._load_dataset()

        # Use first and last state as init and goal
        self.init_pos = self.observations[0][:2]
        self.goal = self.goals[0]
        self.max_steps = 3000

        # Define observation and action space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Maze2D compatibility
        self._target = self.goal.copy()
        self._max_episode_steps = self.max_steps
        self.max_episode_steps = self.max_steps

        # Maze rendering (optional)
        self.maze_arr = np.zeros((100, 100), dtype=np.int32)
        self.maze_arr[20:80, 20:30] = 1
        self.maze_arr[70:80, 20:70] = 1
        goal_x, goal_y = np.clip(self.goal.astype(int), 0, 99)
        self.maze_arr[goal_x, goal_y] = 10

        # Internal state
        self.state = None
        self.step_count = 0

        utils.EzPickle.__init__(self)

    def _load_dataset(self):
        with h5py.File(self.asset_path, 'r') as f:
            self.observations = f['observations'][:]
            self.actions = f['actions'][:]
            self.next_observations = f['next_observations'][:]
            self.rewards = f['rewards'][:]
            self.terminals = f['terminals'][:]
            self.goals = f['goals'][:]


    def _is_valid(self, x, y):
        # Check bounds
        if x < 0 or x >= self.maze_arr.shape[0] or y < 0 or y >= self.maze_arr.shape[1]:
            return False
        # Check walls (maze_arr == 1 means wall)
        if self.maze_arr[int(x), int(y)] == 1:
            return False
        return True

    # def reset(self):
    #     self.state = np.concatenate([
    #         self.init_pos.copy(),               # x, y
    #         np.array([0.0]),                    # theta
    #         np.zeros(2, dtype=np.float32)       # v, omega
    #     ])
    #     self.step_count = 0
    #     return self._get_obs()

    def reset(self):
        idx = np.random.randint(len(self.observations))
        self.state = self.observations[idx].copy()
        self.goal = self.goals[idx].copy()
        self.step_count = 0
        return self._get_obs()


    def step(self, action):
        # # x, y, theta, v, omega = self.state
        # x, y, theta = self.state[:3]
        # # dx, dy = action[0] * np.cos(theta), action[0] * np.sin(theta)
        # # new_x = x + dx
        # # new_y = y + dy
        # # new_theta = theta + action[1]
        # dx = action[0]
        # dy = action[1]
        # new_x = x + dx
        # new_y = y + dy
        # new_theta = theta  # keep orientation unchanged


        # unpack state
        x, y, theta, _, _ = self.state

        # interpret action as [v, omega]
        v, omega = action

        # TurtleBot differential-drive motion model
        new_x = x + v * np.cos(theta)
        new_y = y + v * np.sin(theta)
        new_theta = theta + omega

        # Check validity
        if not self._is_valid(new_x, new_y):
            # Option A: terminate with penalty
            reward = -5.0
            done = True
            self.step_count += 1
            return self._get_obs(), reward, done, {"collision": True}

        # If valid, update state
        # self.state = np.array([new_x, new_y, new_theta, action[0], action[1]])
        self.state = np.array([new_x, new_y, new_theta, v, omega], dtype=np.float32)


        # forward_reward = np.linalg.norm([dx, dy])
        forward_reward = abs(v)
        ctrl_cost = 0.5 * np.square(action).sum()
        reward = forward_reward - ctrl_cost + 1.0
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost + survive_reward

        done = False
        if np.linalg.norm(self.state[:2] - self.goal) < 1.0:
            reward += 10
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        self.step_count += 1
        return self._get_obs(), reward, done, {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            # 'reward_survive': survive_reward
        }


    def _get_obs(self):
        return self.state.copy()

    def viewer_setup(self):
        pass

    def get_dataset(self, load_path=None):
        path = load_path or self.asset_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with h5py.File(path, 'r') as f:
            observations = f['observations'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]
            goals = f['goals'][:]
            timeouts = f['timeouts'][:] if 'timeouts' in f else np.zeros_like(terminals)

            next_observations = f['next_observations'][:] if 'next_observations' in f else observations[1:]

            # Filter invalid states
            mask = [self._is_valid(obs[0], obs[1]) for obs in observations]
            observations = observations[mask]
            actions = actions[mask]
            rewards = rewards[mask]
            terminals = terminals[mask]
            next_observations = next_observations[mask]
            goals = goals[mask]

            dataset = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'terminals': terminals,
                'timeouts': timeouts[mask],
                'next_observations': next_observations,
                'goals': goals
            }
        return dataset

    
    def get_normalized_score(self, raw_score):
        min_score = 0
        max_score = 5000
        return (raw_score - min_score) / (max_score - min_score)

    def render(self, mode="human"):
        plt.imshow(self.maze_arr.T, origin="lower", cmap="gray")
        x, y = self.state[:2]
        plt.scatter(x, y, c="blue")
        gx, gy = self.goal
        plt.scatter(gx, gy, c="red")
        plt.show()
