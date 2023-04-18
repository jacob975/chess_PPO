from gym_chess import GymChessEnv
from sb3_contrib import MaskablePPO
import numpy as np
import chess

# Instantiate the env
env = GymChessEnv()
adversary = MaskablePPO.load("last_model", env=env, verbose=1)
env.set_adversary(adversary)

# Play the game with the adversary
env.env.reset()
print(env.render())
done = False
while not done:
    # Show all possible actions
    action_mask = env.action_mask
    possible_actions = np.where(action_mask == 1)[0]
    # Integer to UCI
    possible_moves = [env.action_to_move(action) for action in possible_actions]
    print(possible_moves)
    # Ask the user to input a valid action
    while True:
        chosen_move = input("Next move: ")
        if chosen_move in possible_moves:
            break
        else:
            print("Invalid move. Try again.")
    chosen_action = env.move_to_action(chosen_move)
    print("Chosen action: ", chosen_action)
    # Play the action
    obs, reward, done, info = env.step(chosen_action)
    # Show the board
    print(env.render())
    print(f"reward: {reward}, done: {done}, info: {info}")
