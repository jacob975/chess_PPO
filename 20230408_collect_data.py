from pettingzoo.classic import chess_v5
import numpy as np

env = chess_v5.env(render_mode='ansi')
env.reset()

# Collect observations and action masks from the env
observations = []
action_masks = []
counter = 0
while counter < 1000000:
    print(counter)
    for i in range(1000000):
        observations.append(env.observe(f'player_{i%2}')['observation'])
        action_masks.append(env.observe(f'player_{i%2}')['action_mask'])
        counter += 1

        # Whether the game is done for the user who just input action
        done = env.terminations[f'player_{i%2}']
        if done:
            env.reset()
            break

        action = int(np.random.choice(np.where(action_masks[-1]>0)[0]))
        env.step(action)

# Print the shape
print("The shape of obs is: ", np.array(observations).shape)
print("The shape of action_masks is: ", np.array(action_masks).shape)
# Save the observations and action masks
np.save('./datasets/observations.npy', np.array(observations))
np.save('./datasets/action_masks.npy', np.array(action_masks))