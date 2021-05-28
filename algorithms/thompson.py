import numpy as np
from algorithms.synthetic_base import SyntheticBase

class Thompson(SyntheticBase):
    def __init__(self,
                 n_arms,
                 n_features,
                 alpha=1.,
                 prior_multiplier=1.,
                ):
        super().__init__(n_arms=n_arms,
                         n_features=n_features,
                         alpha=alpha,
                         prior_multiplier=prior_multiplier)
    
    def decision(self, current_contexts):
        theta = np.random.multivariate_normal(mean=self.theta, cov = self.alpha ** 2 * np.linalg.inv(self.A), size=1)
        action_values = np.matmul(current_contexts, theta.T).squeeze()
        return np.argmax(action_values)