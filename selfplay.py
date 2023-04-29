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

policy_kwargs = dict(
    features_extractor_class=TransformerModel,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)
adversary = MaskablePPO(
    "MlpPolicy", env, gamma=0.99, seed=None, verbose=1,
    device="cpu",
    policy_kwargs=policy_kwargs,
)
adversary = MaskablePPO.load("transformer-batch4k-clip02-dropout02-env4-epoch5-ppo-gamma099/last_model", env=env, verbose=1)
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
    print(f"reward: {env.env.rewards[f'player_{env.turn}']}, done: {done}")
    if done:
        env.reset() # Reset again if the game is done after a random action
        break
    env.turn = (env.turn+1) % 2
    time.sleep(1)