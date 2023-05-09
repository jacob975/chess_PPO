from pettingzoo.classic import chess_v5
import numpy as np
env = chess_v5.env(render_mode='ansi')
env.reset()
print(env.render())
action_mask = env.observe(f'player_0')['action_mask']
observation = env.observe(f'player_0')['observation']
table = observation[:,:,7] # The pawns
print("In the view of Player 0")
print(table)
possible_actions = np.where(action_mask>0)[0]
print(possible_actions)
action = 77
print(action)
env.env.step(action)
# After an action
print("------------------")
print(env.render())
observation = env.observe(f'player_0')['observation']
table = observation[:,:,7]
print("In the view of Player 0")
print(table)
observation = env.observe(f'player_1')['observation']
table = observation[:,:,7]
print("In the view of Player 1")
print(table)