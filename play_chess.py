from gym_chess import GymChessEnv
from sb3_contrib import MaskablePPO
import numpy as np
import chess
from custom_policy import CustomCNN

# Instantiate the env
env = GymChessEnv()
env.env.reset()
env.turn = 0

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)
adversary = MaskablePPO(
    "MlpPolicy", env, gamma=0.99, seed=None, verbose=1,
    device="cpu",
    policy_kwargs=policy_kwargs,
)
adversary = MaskablePPO.load("last_model", env=env, verbose=1)
adversary.policy.features_extractor.training = False
env.set_adversary(adversary)

# Play the game with the adversary

print(env.render())
done = False
while not done:
    # Show all possible actions
    action_mask = env.observe(f"player_{env.turn}")["action_mask"]
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
