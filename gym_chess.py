from pettingzoo.classic import chess_v5
import gym
import numpy as np

# Use gym to wrap the PettingZoo environment
class GymChessEnv(gym.Env):
    def __init__(self, **kwargs):
        self.env = chess_v5.env(render_mode='ansi', **kwargs)
        self.turn = 0
        self.reset()

    def reset(self):
        self.env.reset()
        self.done = False
        self.turn = 0
        self.obs = self.env.observe(f'player_{self.turn % 2}')
        # Define the action space and cast it to gym.spaces.discrete.Discrete
        action_space = self.env.action_space(f'player_{self.turn % 2}')
        self.action_space = gym.spaces.discrete.Discrete(action_space.n)
        # DEBUG
        observation_space = self.env.observe(f'player_{self.turn % 2}')['observation']
        self.observation_space = gym.spaces.box.Box(low=0, high=1, shape=observation_space.shape, dtype=np.float32)
        self.action_mask = self.env.observe(f'player_{self.turn % 2}')['action_mask']
        return self.obs

    def step(self, action):
        self.env.step(action)
        done = self.env.terminations[f'player_{self.turn % 2}']
        reward = self.env.rewards[f'player_{self.turn % 2}']
        info = self.env.infos[f'player_{self.turn % 2}']
        self.turn += 1
        obs = self.env.observe(f'player_{self.turn % 2}')
        self.action_mask = self.env.observe(f'player_{self.turn % 2}')['action_mask']
        return obs, reward, done, info

    def render(self):
        return self.env.render()