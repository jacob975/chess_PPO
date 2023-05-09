from pettingzoo.classic import chess_v5
from sb3_contrib import MaskablePPO
from stable_baselines3.ppo import PPO
# Import chess_utils from pettingzoo
from pettingzoo.classic.chess import chess_utils

import gym
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks


# Use gym to wrap the PettingZoo environment
class GymChessEnv(gym.Env):
    def __init__(self, **kwargs):
        self.env = chess_v5.env(render_mode='ansi', **kwargs)
        self.adversary = None
        self.reset()

    def reset(self):
        # Reset the environment
        self.env.reset()
        self.turn = 0

        # The environment was set to a random state. i.e. all pieces are randomly placed on the board.
        pre_steps = np.random.randint(2)
        for i in range(pre_steps):
            # Adversary
            if isinstance(self.adversary, MaskablePPO):
                # Get action from the adversary
                action_mask = self.observe(f'player_{self.turn}')['action_mask']
                action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'], action_masks=action_mask)[0])
            elif isinstance(self.adversary, PPO):
                # Get action from the adversary
                action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'])[0])
            else:
                # If the adversary is not set, sample a random action from action_mask
                action_mask = self.observe(f'player_{self.turn}')['action_mask']
                possible_actions = np.where(action_mask>0)[0]
                action = int(np.random.choice(possible_actions))

            ## Random actions
            #action_mask = self.observe(f'player_{self.turn}')['action_mask']
            #possible_actions = np.where(action_mask>0)[0]
            #action = int(np.random.choice(possible_actions))

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

    def action_to_move(self, action:int):
        # Convert action to UCI
        # e.g. action = 0, uci = 'a1a2'
        # e.g. action = 1, uci = 'a1a3'
        # e.g. action = 2, uci = 'a1a4'
        uci = chess_utils.actions_to_moves[action]
        return uci
    
    def move_to_action(self, uci:str):
        # Convert UCI to action
        # e.g. uci = 'a1a2', action = 0
        # e.g. uci = 'a1a3', action = 1
        # e.g. uci = 'a1a4', action = 2
        action = chess_utils.moves_to_actions[uci]
        return action

    def step(self, action):
        # DO action
        self.env.step(action)
        # Whether the game is done for the user who just input action
        done = self.env.terminations[f'player_{self.turn}']
        # Reward for input action
        reward = self.env.rewards[f'player_{self.turn}']
        # FEN as info
        info = self.env.infos[f'player_{self.turn}']
        #info = self.env.env.board.fen()
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
        
        if isinstance(self.adversary, MaskablePPO):
            # Get action from the adversary
            action_mask = self.observe(f'player_{self.turn}')['action_mask']
            action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'], action_masks=action_mask)[0])
        elif isinstance(self.adversary, PPO):
            # Get action from the adversary
            action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'])[0])
        else:
            # If the adversary is not set, sample a random action from action_mask
            action_mask = self.observe(f'player_{self.turn}')['action_mask']
            possible_actions = np.where(action_mask>0)[0]
            action = int(np.random.choice(possible_actions))

        # Do the adversarial action
        self.env.step(action)
        self.turn = (self.turn+1) % 2
        # Whether the game is done for the user who just input action
        done = self.env.terminations[f'player_{self.turn}']
        # Reward for input action
        reward = self.env.rewards[f'player_{self.turn}']
        # FEN as info
        info = self.env.infos[f'player_{self.turn}']
        #info = self.env.env.board.fen()
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
        # Take the first 20 channels from a 8x8x111 table
        # and convert it to a 20x8x8 table
        context['observation'] = np.transpose(context['observation'], (2, 0, 1))
        #context['observation'] = context['observation'][:20]
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
    
    def action_masks(self):
        return self.action_mask
    
    # Estimate the win rate of the agent against the adversary
    # If the adversary is None, then the agent plays against a the set adversary
    def estimate_winrate(self, agent, adversary=None, runs:int=10):
        if adversary is None:
            adversary = self.adversary
        
        # Play 20 games to estimate the win rate
        agent_wins = 0
        adversary_wins = 0
        draws = 0
        for i in range(runs):
            # Reset the environment
            self.env.reset()
            self.turn = 0
            action_chain = []

            who_starts = np.random.randint(2)
            if who_starts == 1: # Adversary move first as turn 0
                # Adversary's turn
                if isinstance(self.adversary, MaskablePPO):
                    # Get action from the adversary
                    action_mask = self.observe(f'player_{self.turn}')['action_mask']
                    action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'], action_masks=action_mask)[0])
                elif isinstance(self.adversary, PPO):
                    # Get action from the adversary
                    action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'])[0])
                else:
                    # If the adversary is not set, sample a random action from action_mask
                    action_mask = self.observe(f'player_{self.turn}')['action_mask']
                    possible_actions = np.where(action_mask>0)[0]
                    action = int(np.random.choice(possible_actions))
                self.env.step(action)
                done = self.env.terminations[f'player_{self.turn}']
                if done:
                    #print(self.env.rewards[f'player_{(self.turn+1) % 2}'], self.env.rewards[f'player_{self.turn}'])
                    if self.env.rewards[f'player_{self.turn}'] == 1: adversary_wins += 1
                    elif self.env.rewards[f'player_{self.turn}'] == -1: agent_wins += 1
                    else: draws += 1
                    continue
                self.turn = (self.turn+1) % 2

            # Start the game
            while True:

                # Agent's turn
                if isinstance(agent, MaskablePPO):
                    action_mask = self.observe(f'player_{self.turn}')['action_mask']
                    action = int(agent.predict(self.observe(f'player_{self.turn}')['observation'], action_masks=action_mask)[0])
                elif isinstance(agent, PPO):
                    action = int(agent.predict(self.observe(f'player_{self.turn}')['observation'])[0])
                action_chain.append(action)
                self.env.step(action)
                done = self.env.terminations[f'player_{self.turn}']
                if done:
                    #print(self.env.rewards[f'player_{self.turn}'], self.env.rewards[f'player_{(self.turn+1) % 2}'])
                    if self.env.rewards[f'player_{self.turn}'] == 1: agent_wins += 1
                    elif self.env.rewards[f'player_{self.turn}'] == -1: adversary_wins += 1
                    else: draws += 1
                    break
                self.turn = (self.turn+1) % 2

                # Adversary's turn
                if isinstance(self.adversary, MaskablePPO):
                    # Get action from the adversary
                    action_mask = self.observe(f'player_{self.turn}')['action_mask']
                    action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'], action_masks=action_mask)[0])
                elif isinstance(self.adversary, PPO):
                    # Get action from the adversary
                    action = int(self.adversary.predict(self.observe(f'player_{self.turn}')['observation'])[0])
                else:
                    # If the adversary is not set, sample a random action from action_mask
                    action_mask = self.observe(f'player_{self.turn}')['action_mask']
                    possible_actions = np.where(action_mask>0)[0]
                    action = int(np.random.choice(possible_actions))                
                self.env.step(action)
                done = self.env.terminations[f'player_{self.turn}']
                if done:
                    #print(self.env.rewards[f'player_{(self.turn+1) % 2}'], self.env.rewards[f'player_{self.turn}'])
                    if self.env.rewards[f'player_{self.turn}'] == 1: adversary_wins += 1
                    elif self.env.rewards[f'player_{self.turn}'] == -1: agent_wins += 1
                    else: draws += 1
                    break
                self.turn = (self.turn+1) % 2
            #print("Action chain: ", action_chain)

        # Print to DEBUG
        print(f"Agent wins: {agent_wins}, Adversary wins: {adversary_wins}, Draws: {draws}")
        # Reset the environment
        self.reset()

        return agent_wins/runs