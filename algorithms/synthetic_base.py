import numpy as np

class SyntheticBase:
    def __init__(self,
                 n_arms,
                 n_features,
                 alpha=1.,
                 prior_multiplier=1.,
                ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.prior_multiplier = prior_multiplier
        
        self.reset()
    
    
    def reset(self):
        # record iteration
        self.iteration = 0
        self.A = np.eye(self.n_features) * self.prior_multiplier
        self.cum_r = np.zeros(self.n_features)
        self.theta = np.ones(self.n_features) / self.n_arms
        
    def decision(self, current_contexts):
        raise NotImplemented('Should give how to make decisions.')
        
    def update(self, action, context, reward):
        self.A = self.A + np.matmul(context[action].reshape(-1, 1), context[action].reshape(1, -1))
        self.cum_r = self.cum_r + reward * context[action]
        self.theta = np.matmul(np.linalg.inv(self.A), self.cum_r)
        