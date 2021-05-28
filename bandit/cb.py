import numpy as np

class CB:
    def __init__(self,
                 name,
                 n_arms,
                 contexts_shape):
        self.name = name
        self.n_arms = n_arms
        self.contexts_shape = contexts_shape
        self.best_rewards_oracle = np.array([])
        self.iteration = 0
        self.index = 0
        self.all_contexts = np.array([])
        self.reset()
        
    def reset_iteration(self):
        self.iteration = 0
        
    def reset(self):
        raise NotImplemented('Should implement reset for different settings.')
        
    def current_regret(self):
        return self.regret
        
    def get_next_context(self):
        return self.contexts[self.index]
    
    def _get_reward(self, iteration, action, expectation=False):
        return self.rewards[iteration, action] + np.random.normal(0., 1.) * self.reward_noise_scale
    
    def step(self, action):
        self.regret = self.best_rewards_oracle[self.index] - self._get_reward(self.index, action, expectation=True)
        current_reward = self._get_reward(self.index, action, expectation=False)
        self.iteration += 1
        self.index = np.random.randint(0, self.best_rewards_oracle.shape[0])
        return current_reward
    
    def get_all_contexts(self):
        return self.contexts
    
    def get_all_rewards(self):
        return self.rewards