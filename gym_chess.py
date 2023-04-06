from pettingzoo.classic import chess_v5
import gym
import numpy as np

# Use gym to wrap the PettingZoo environment
class GymChessEnv(gym.Env):
    def __init__(self, **kwargs):
        self.env = chess_v5.env(render_mode='ansi', **kwargs)
        self.turn = 0
        self.adversary = None
        self.reset()

    def reset(self):
        # Reset the environment
        self.env.reset()
        self.done = False
        self.turn = 0

        # The environment was set to a random state. i.e. all pieces are randomly placed on the board.
        len_random_actions = np.random.randint(0, 40)
        for i in range(len_random_actions):
            action_mask = self.observe(f'player_{self.turn}')['action_mask']
            possible_actions = np.where(action_mask>0)[0]
            action = int(np.random.choice(possible_actions))
            self.env.step(action)
            done = self.env.terminations[f'player_{self.turn}']
            if done:
                self.reset() # Reset again if the game is done after a random action
                break
            self.turn = (self.turn+1) % 2

        # Define the action space and cast it to gym.spaces.discrete.Discrete
        action_space = self.env.action_space(f'player_{self.turn}')
        self.action_space = gym.spaces.discrete.Discrete(action_space.n)
        obs = self.observe(f'player_{self.turn}')['observation']
        self.observation_space = gym.spaces.box.Box(low=0, high=1, shape=obs.shape, dtype=np.float32)
        self.action_mask = self.observe(f'player_{self.turn}')['action_mask']
        return obs

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
        obs = self.observe(f'player_{self.turn}')['observation']
        # DEBUG
        #print("The player's turn is: ", self.turn)
        #print(obs[14])
        #print(obs.shape)
        #print(self.env.render())
        self.action_mask = self.observe(f'player_{self.turn}')['action_mask']
        if done:
            return obs, reward, done, info
        
        # Adversary
        if self.adversary is not None:
            # Get action from adversary
            action = int(self.adversary(obs)[0])
        else:
            # If the adversary is not set, sample a random action from action_mask
            possible_actions = np.where(self.action_mask>0)[0]
            action = int(np.random.choice(possible_actions))

        # Do the adversarial action
        self.env.step(action)
        self.turn = (self.turn+1) % 2
        # Whether the game is done for the user who just input action
        done = self.env.terminations[f'player_{self.turn}']
        # Reward for input action
        reward = self.env.rewards[f'player_{self.turn}']
        # Info for input action
        info = self.env.infos[f'player_{self.turn}']
        # Observation for the next user
        obs = self.observe(f'player_{self.turn}')['observation']
        # DEBUG
        #print("The player's turn is: ", self.turn)
        #print(obs[14])
        #print(obs.shape)
        #print(self.env.render())
        self.action_mask = self.observe(f'player_{self.turn}')['action_mask']

        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def set_adversary(self, adversary):
        self.adversary = adversary

    def observe(self, agent):
        context = self.env.observe(agent)
        # Convert take the first 20 channels from a 8x8x111 table
        # and convert it to a 20x8x8 table
        context['observation'] = np.transpose(context['observation'], (2, 0, 1))
        # If self.turn == 0, then the agent is white and the adversary is black
        # If self.turn == 1, then the agent is black and the adversary is white
        # channels 0-5 are white pieces and channels 6-11 are black pieces, 12 is empty
        # So we need to swap the first 6 channels with the 6-11 channels
        if self.turn == 1:
            # Section 1: Transpose the board
            context['observation'] = np.flip(context['observation'], axis=1)
            context['observation'] = np.flip(context['observation'], axis=2)
            # Section 2: Swap the first 6 channels with the last 6 channels
            for i in range(1,9):
                tmp = context['observation'][13*i-6 : 13*i].copy()
                context['observation'][13*i-6 : 13*i] = context['observation'][13*i : 13*i+6]
                context['observation'][13*i : 13*i+6] = tmp
        return context