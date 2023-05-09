from gym_chess import GymChessEnv
from sb3_contrib import MaskablePPO
import numpy as np
import chess
from custom_policy import CustomCNN, TransformerModel
import time

# Instantiate the env
env = GymChessEnv()
#env.env.reset()
#env.turn = 0

dropout = 0.1
nlayers = 1
d_model = 512
d_hid = 2048

policy_kwargs = dict(
    features_extractor_class=TransformerModel,
    features_extractor_kwargs=dict(features_dim=4672, dropout=dropout, nlayers=nlayers, d_model=d_model, d_hid = d_hid),
)
adversary = MaskablePPO(
    "MlpPolicy", env, gamma=0.99, seed=None, verbose=1,
    policy_kwargs=policy_kwargs,
    device="cpu",
)
adversary = MaskablePPO.load("transformer-nlayer4-batch4k-clip02-dropout01-env4-epoch5-ppo-gamma099/last_model", env=env, verbose=1)
adversary.policy.features_extractor.training = False
env.set_adversary(adversary)

# Play the game with the adversary

print(env.render())
done = False
while not done:
    # Random actions
    action_mask = env.observe(f'player_{env.turn}')['action_mask']
    action = int(adversary.predict(env.observe(f'player_{env.turn}')['observation'], action_masks=action_mask)[0])
    env.env.step(action)
    done = env.env.terminations[f'player_{env.turn}']
    print(env.render())
    print(f"player 0: {env.env.rewards[f'player_0']}, player 1: {env.env.rewards[f'player_1']}, done: {done}")
    if done:
        env.reset() # Reset again if the game is done after a random action
        break
    env.turn = (env.turn+1) % 2
    time.sleep(1)