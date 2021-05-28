import numpy as np
import torch
import math
import copy
import torch.nn.functional as F
from algorithms.base import Base
from algorithms.models.utils import build_model

class ROFU(Base):
    def __init__(self,
                 writer,
                 model_config,
                 lr_init=0.01,
                 lr_decay = 1.,
                 coef=1., 
                 batch_size=-1,
                 learning_rate=0.01,
                 train_lr_decay=1.,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                 rofu_epoch=1,
                 reg_constant=1.,
                 last_layer_feature=0,
                 optimizer_name='adam'
                ):
        self.lr_decay = lr_decay
        self.coef = coef
        self.rofu_epoch = rofu_epoch
        self.reg_constant = reg_constant
        self.last_layer_feature = last_layer_feature
        self.arm_pulled_times = np.zeros(n_arms)
        self.window_size = 10
        self.optimizer_name = optimizer_name
        self.avg_loss = np.ones(self.window_size)
        self.lr_init=lr_init
        super().__init__(writer=writer,
                         model_config=model_config,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         train_lr_decay=train_lr_decay,
                         train_data_epoch=train_data_epoch,
                         shared_parameter=shared_parameter,
                         loss_type=loss_type,
                         n_arms=n_arms)
        
    def _train_on_action(self, contexts):
        if self.actions.size == 0:
            return np.random.random(self.n_arms)
        action_values = np.zeros(self.n_arms)
        before_training_model = copy.deepcopy(self.model.state_dict())
        before_training_opt = copy.deepcopy(self.optimizer.state_dict())
        for action in range(self.n_arms):
            new_model = copy.deepcopy(self.model)
            new_model.train()
            if self.optimizer_name == 'sgd':
                new_opt = torch.optim.SGD(new_model.parameters(), lr=self.lr_init * self.lr_decay ** self.iteration)
            else:
                new_opt = torch.optim.RMSprop(new_model.parameters(), lr=0.0001)
            for epoch in range(self.rofu_epoch):
                x_train, y_train, actions_so_far = self._sample_data()
                if self.loss_type == 'mse':
                    y_pred = new_model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far].squeeze()
                else:
                    y_pred = torch.exp(new_model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far]).squeeze()
        
                if self.loss_type == 'mse':
                    current_value = new_model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).squeeze()[action]
                else:
                    current_value = torch.exp(new_model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).squeeze()[action])


                loss = -current_value / (self.reg_constant * (self.arm_pulled_times[action] + 1)) + self.loss_func(y_pred, y_train)
                new_opt.zero_grad()
                loss.backward()
                new_opt.step()
            new_model.eval()
            action_values[action] = new_model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()[action]
            self.model.load_state_dict(before_training_model)
            self.optimizer.load_state_dict(before_training_opt)
        return action_values
        
    def update(self, action, context, reward):
        self.actions_list.append(action)
        if self.shared_parameter:
            self.contexts_list.append(context[action])
        else:
            self.contexts_list.append(context)
        self.rewards_list.append(reward)
        self.actions = np.array(self.actions_list)
        self.contexts = np.array(self.contexts_list)
        self.rewards = np.array(self.rewards_list)
        self.iteration += 1

        
    def decision(self, current_contexts):
        loss = self._train_on_data()
        self.model.eval()
        _current_contexts = torch.FloatTensor(current_contexts).to(self.device).unsqueeze(dim=0)
        mean_reward = self.model.forward(_current_contexts).cpu().detach().squeeze().numpy()
        action_values = self._train_on_action(current_contexts)
        self.mean_reward = mean_reward
        self.action_values = action_values
        gap = np.maximum(np.zeros(self.n_arms), self.action_values - self.mean_reward)
        self.ucb = self.mean_reward + self.coef * np.sqrt(gap)

        self.writer.add_scalar("ucb", np.mean(gap), self.iteration)
        self.writer.add_scalar("ucb_max", np.max(gap), self.iteration)
        self.writer.add_scalar("ucb_min", np.min(gap), self.iteration)

        pulled_arm = np.argmax(self.ucb)
        self.arm_pulled_times[pulled_arm] += 1
        return pulled_arm