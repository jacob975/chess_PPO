from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SetAdversaryCallback(BaseCallback):
    def __init__(self, update_freq, adversary, model_name = "last_model", adversary_name = "adversary_model"):
        super(SetAdversaryCallback, self).__init__()
        self.update_freq = update_freq
        # THe adversary should have the same type as the model
        #assert type(adversary) == type(self.model)
        self.adversary = adversary
        self.best_winrate = 0
        self.best_winrate_no_improvement = 0 # Number of times the winrate has not improved
        self.model_name = model_name
        self.adversary_name = adversary_name

    def _init_callback(self) -> None:
        #if self.adversary is None:
        #    print("No adversary set")
        #else:
        #    self.training_env.env_method("set_adversary", self.adversary)
        #    print("Adversary set")
        return super()._init_callback()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            #agent_winrate = np.mean(self.training_env.env_method("estimate_winrate", agent = self.model))
            ep_rew_matrix = [rew_list[-10:] for rew_list in self.training_env.env_method("get_episode_rewards")]
            ep_rew_mean = np.mean([rew_list.count(1)/len(rew_list) for rew_list in ep_rew_matrix])
            str_winrate = f"Agent winrate: {100*ep_rew_mean:.2f} %."
            # Update the adversary if the winrate of the agent is higher than 0.55
            if ep_rew_mean >= 0.55:
                self.model.save(self.adversary_name)
                self.adversary.set_parameters(self.model.get_parameters())
                self.training_env.env_method("set_adversary", self.adversary)
                str_update = "Adversary updated"
            else:
                str_update = "Adversary not updated"

            # Save another model anyway
            self.model.save(self.model_name)
            
            print(str_winrate, str_update)
            
        return True