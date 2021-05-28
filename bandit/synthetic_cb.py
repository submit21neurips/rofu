import numpy as np
import torch
import itertools
from bandit.cb import CB
from algorithms.models.utils import build_model
from algorithms.models.utils import Model
class SyntheticCB(CB):
    def __init__(self,
                 name,
                 n_arms,
                 contexts_shape,
                 timestep=1000,
                 mode='linear',
                 same=False,
                 noise_std=1.,
                 feature=None,
                ):
        self.timestep = timestep
        self.mode = mode
        self.same = same
        self.noise_std = noise_std
        self.feature = feature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(name,
                         n_arms,
                         contexts_shape)
        
    def reset(self):
        if self.mode == 'resnet':
            self.model = build_model({'type': 'resnet20'}, self.device)
            self.contexts = np.random.randint(-5, 5, (self.timestep,) + self.contexts_shape).astype(float)
            state_dict = torch.load('./algorithms/models/pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th')

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            #load params
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            self.true_rewards = self.model.forward(torch.FloatTensor(self.contexts).to(self.device)).cpu().detach().squeeze().numpy()
            self.rewards = np.array([self.true_rewards[t, k] + np.random.normal() * self.noise_std for t, k in itertools.product(range(self.timestep), range(10))]).reshape(self.timestep, 10)
            
            self.best_actions_oracle = np.argmax(self.true_rewards, axis=1)
            self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.timestep)])
        elif self.mode == 'linear':
            if self.feature.size == 0:
                self.feature = np.random.randn(*self.contexts_shape)
            self.contexts = np.random.randn(self.timestep, self.n_arms, *self.contexts_shape)
            self.all_contexts = self.contexts
            self.true_rewards = np.array(
                [
                    np.dot(self.feature, self.contexts[t, k]) \
                    for t,k in itertools.product(range(self.timestep), range(self.n_arms))
                ]).reshape(self.timestep, self.n_arms)

            self.rewards = np.array(
                [
                    self.true_rewards[t, k] + self.noise_std*np.random.randn()\
                    for t,k in itertools.product(range(self.timestep), range(self.n_arms))
                ]
            ).reshape(self.timestep, self.n_arms)
            self.best_actions_oracle = np.argmax(self.true_rewards, axis=1)
            self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.timestep)])
        elif self.mode == 'mlp':
            self.model = Model(input_size=self.contexts_shape[0],
                      hidden_size=20,
                      n_layers=2,
                      activation='ReLU',
                      p=0.0,
                      initialization=None,
                      output_size=self.n_arms).to(self.device)
            self.contexts = 0.1 * np.random.random((self.timestep,self.contexts_shape[0]) )
            # state_dict = torch.load('./algorithms/models/pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th')

            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict['state_dict'].items():
            #     name = k[7:] # remove `module.`
            #     new_state_dict[name] = v
            # # load params
            # self.model.load_state_dict(new_state_dict)
            self.model.eval()
            self.true_rewards = self.model.forward(torch.FloatTensor(self.contexts).to(self.device)).cpu().detach().squeeze().numpy()
            self.rewards = np.array([self.true_rewards[t, k] + np.random.normal() * self.noise_std for t, k in itertools.product(range(self.timestep), range(10))]).reshape(self.timestep, 10)
            
            self.best_actions_oracle = np.argmax(self.true_rewards, axis=1)
            self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.timestep)])


    def _get_reward(self, iteration, action, expectation=False):
        if expectation:
            return self.true_rewards[iteration, action]
        #print("reward", self.true_rewards[iteration, action])
        #print("reward", self.rewards[iteration, action])
        return self.rewards[iteration, action]