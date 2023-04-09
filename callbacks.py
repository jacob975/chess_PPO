from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SetAdversaryCallback(BaseCallback):
    def __init__(self, update_freq, adversary):
        super(SetAdversaryCallback, self).__init__()
        self.update_freq = update_freq
        # THe adversary should have the same type as the model
        #assert type(adversary) == type(self.model)
        self.adversary = adversary

    def _init_callback(self) -> None:
        if self.adversary is None:
            print("No adversary set")
        else:
            self.training_env.env_method("set_adversary", self.adversary.predict)
            print("Adversary set")
        return super()._init_callback()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0 and self.adversary is not None:
            agent_winrate = np.mean(self.training_env.env_method("estimate_winrate", agent = self.model.predict, adversary = self.adversary.predict))
            print("Agent winrate: ", f"{100*agent_winrate:.2f} %")
            # Update the adversary if the winrate of the agent is higher than 0.55
            if agent_winrate >= 0.55:
                self.adversary.set_parameters(self.model.get_parameters())
                print("Adversary updated")
            else:
                print("Adversary not updated")
        return True