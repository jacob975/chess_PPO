from stable_baselines3.common.callbacks import BaseCallback

class SetAdversaryCallback(BaseCallback):
    def __init__(self, update_freq, adversary):
        super(SetAdversaryCallback, self).__init__()
        self.update_freq = update_freq
        # THe adversary should have the same type as the model
        assert type(adversary) == type(self.model)
        self.adversary = adversary

    def _init_callback(self) -> None:
        self.training_env.env_method("set_adversary", self.adversary.predict)
        print("Adversary set")
        return super()._init_callback()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # Set the parameters of the adversary
            self.adversary.set_parameters(self.model.get_parameters())
            print("Adversary updated")
        return True