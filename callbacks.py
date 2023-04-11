from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SetAdversaryCallback(BaseCallback):
    def __init__(self, update_freq, adversary):
        super(SetAdversaryCallback, self).__init__()
        self.update_freq = update_freq
        # THe adversary should have the same type as the model
        #assert type(adversary) == type(self.model)
        self.adversary = adversary
        self.best_winrate = 0
        self.best_winrate_no_improvement = 0 # Number of times the winrate has not improved

    def _init_callback(self) -> None:
        #if self.adversary is None:
        #    print("No adversary set")
        #else:
        #    self.training_env.env_method("set_adversary", self.adversary)
        #    print("Adversary set")
        return super()._init_callback()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            agent_winrate = np.mean(self.training_env.env_method("estimate_winrate", agent = self.model))
            str_winrate = f"Agent winrate: {100*agent_winrate:.2f} %."
            # Update the adversary if the winrate of the agent is higher than 0.55
            if agent_winrate >= 0.55:
                self.adversary.set_parameters(self.model.get_parameters())
                self.training_env.env_method("set_adversary", self.adversary)
                str_update = "Adversary updated"
            else:
                str_update = "Adversary not updated"

            if agent_winrate > self.best_winrate:
                self.best_winrate = agent_winrate
                self.model.save("best_model")
                str_update += " and model saved"
                self.best_winrate_no_improvement = 0
                print(str_winrate, str_update)

            # Stop the training if the best_winrate has no improvement for 5 times
            #elif agent_winrate < self.best_winrate:
            #    self.best_winrate_no_improvement += 1
            #    if self.best_winrate_no_improvement >= 5:
            #        self.training_env.env_method("set_adversary", None)
            #        str_update += " and training stopped"
            #        print(str_winrate, str_update)
            #        self._on_training_end()
            #        return False
            
        return True