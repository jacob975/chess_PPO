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
        # Define the action space and cast it to gym.spaces.discrete.Discrete
        action_space = self.env.action_space(f'player_{self.turn}')
        self.action_space = gym.spaces.discrete.Discrete(action_space.n)
        # DEBUG
        self.obs = self.env.observe(f'player_{self.turn}')['observation']
        self.observation_space = gym.spaces.box.Box(low=0, high=1, shape=self.obs.shape, dtype=np.float32)
        self.action_mask = self.env.observe(f'player_{self.turn}')['action_mask']
        return self.obs

    def step(self, action):
        # DO action
        self.env.step(action)
        # Whether the game is done for the user who just input action
        done = self.env.terminations[f'player_{self.turn}']
        # Reward for input action
        reward = self.env.rewards[f'player_{self.turn}']
        # Info for input action
        info = self.env.infos[f'player_{self.turn}']
        self.turn = (self.turn+1) % 2
        # Observation for the next user
        obs = self.env.observe(f'player_{self.turn}')['observation']
        self.action_mask = self.env.observe(f'player_{self.turn}')['action_mask']
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def set_adversary(self, adversary):
        self.adversary = adversary