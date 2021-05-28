import numpy as np
import torch
import math
import copy
import torch.nn.functional as F
from algorithms.base import Base
from algorithms.models.utils import build_model

class ROFU1_2(Base):
    def __init__(self,
                 writer,
                 model_config,
                 coef = 1.,
                 batch_size=-1,
                 learning_rate=0.01,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                 rofu_epoch=1,
                 reg_constant=1.,
                 last_layer_feature=0,
                 optimizer_name='adam'
                ):
        self.coef = coef
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rofu_epoch = rofu_epoch
        self.reg_constant = reg_constant
        self.last_layer_feature = last_layer_feature
        self.arm_pulled_times = np.zeros(n_arms)
        self.window_size = 10
        self.optimizer_name = optimizer_name
        self.avg_loss = np.ones(self.window_size)
        
        self.ucb_model = build_model(model_config, self.device)
        self.ucb_optimizer = torch.optim.Adam(self.ucb_model.parameters(), lr=learning_rate)
        self.ucb_context_list = []
        self.ucb_reward_list = []
        self.ucb_action_list = []

        self.update_timesteps = set()
        for i in range(5000):
            self.update_timesteps.add(int(i*i))
        super().__init__(writer=writer,
                         model_config=model_config,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         train_data_epoch=train_data_epoch,
                         shared_parameter=shared_parameter,
                         loss_type=loss_type,
                         n_arms=n_arms)
        
    def _train_on_action(self, contexts, epoch, _action=None):
        if self.actions.size == 0:
            return np.random.random(self.n_arms)
        before_training_model = copy.deepcopy(self.model.state_dict())
        before_training_opt = copy.deepcopy(self.optimizer.state_dict())

        if _action is None:
            actions = list(range(self.n_arms))
        else:
            actions = [_action]

        action_values = np.zeros(len(actions))


        for action_id in range(len(actions)):
            action = actions[action_id]
            new_model = copy.deepcopy(self.model)
            new_model.train()
            if self.optimizer_name == 'sgd':
                new_opt = torch.optim.SGD(new_model.parameters(), lr=0.01 * 0.999 ** self.iteration)
            elif self.optimizer_name == 'Adam':
                new_opt = torch.optim.Adam(new_model.parameters(), lr=0.001)
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

                print("checking eta", self.reg_constant * self.actions.shape[0], "lr", 0.01 * 0.999 ** self.iteration)
                loss = -current_value / (self.reg_constant * self.actions.shape[0]) + self.loss_func(y_pred, y_train)
                new_opt.zero_grad()
                loss.backward()
                new_opt.step()
            new_model.eval()
            action_values[action_id] = new_model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()[action]
            self.model.load_state_dict(before_training_model)
            self.optimizer.load_state_dict(before_training_opt)

        return action_values

                # for action_id in range(len(actions)):
        #     action = actions[action_id]
        #     for epoch in range(self.rofu_epoch):
        #         x_train, y_train, actions_so_far = self._sample_data()
        #         if self.loss_type == 'mse':
        #             y_pred = self.model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far].squeeze()
        #         else:
        #             y_pred = torch.exp(self.model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far]).squeeze()
        
        #         if self.loss_type == 'mse':
        #             current_value = self.model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).squeeze()[action]
        #         else:
        #             current_value = torch.exp(self.model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).squeeze()[action])


        #         current_reg = math.sqrt(self.arm_pulled_times[action] + 1)
        #         print("checking eta", self.reg_constant * self.actions.shape[0])
        #         loss = -current_value / (self.reg_constant * self.actions.shape[0]) + self.loss_func(y_pred, y_train)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        #     self.model.eval()
        #     action_values[action_id] = self.model.forward(torch.FloatTensor(contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()[action]
        #     self.model.load_state_dict(before_training_model)
        #    self.optimizer.load_state_dict(before_training_opt)



    def _train_ucb_model(self):
        if len(self.contexts_list) < self.batch_size:
            return
        fake_contexts = []
        fake_actions = []
        fake_rewards = []

        leng = int(np.sqrt(self.iteration) + 1)
        for i in range(leng):
            idx = np.random.randint(len(self.contexts_list))
            context = self.contexts_list[idx]
            fake_contexts.append(context)
            #fake_action = np.random.randint(self.n_arms)
            fake_action = self.actions_list[idx]
            fake_actions.append(fake_action)
            all_action_value = self._train_on_action(context, self.rofu_epoch, fake_action)
            assert(len(all_action_value) == 1)
            action_value = all_action_value[0]
            fake_rewards.append(action_value)
        fake_contexts = np.array(fake_contexts)
        fake_actions = np.array(fake_actions)
        fake_rewards = np.array(fake_rewards)
        self.ucb_optimizer = torch.optim.Adam(self.ucb_model.parameters(), lr=0.001)
        for train in range(3 * leng):
            iterations_so_far = np.random.choice(range(leng), self.batch_size, replace=True)
            x_train = torch.FloatTensor(fake_contexts[iterations_so_far]).to(self.device)
            y_train = torch.FloatTensor(fake_rewards[iterations_so_far]).to(self.device)
            actions_so_far = fake_actions[iterations_so_far]
            y_pred = self.ucb_model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far].squeeze()                                    
            loss = self.loss_func(y_train, y_pred)            
            self.ucb_optimizer.zero_grad()
            loss.backward()
            self.ucb_optimizer.step()
        self.writer.add_scalar("ucb_loss", loss, self.iteration)

    


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

        if self.iteration in self.update_timesteps:
            print("updating ucb", self.iteration)
            self._train_ucb_model()
            print("finished update")
        
    def decision(self, current_contexts):
        loss = self._train_on_data()
        self.model.eval()
        _current_contexts = torch.FloatTensor(current_contexts).to(self.device).unsqueeze(dim=0)
        mean_reward = self.model.forward(_current_contexts).cpu().detach().squeeze().numpy()
        optimism_reward = self.ucb_model.forward(_current_contexts).cpu().detach().squeeze().numpy()
        self.mean_reward = mean_reward
        self.action_values = mean_reward

        gap = self.coef * np.sqrt(np.maximum(np.zeros(self.n_arms), optimism_reward - mean_reward))
        ucb = mean_reward + np.sqrt(gap)
        self.writer.add_scalar("ucb", np.mean(gap), self.iteration)

        self.writer.add_scalar("ucb_max", np.max(gap), self.iteration)
        self.writer.add_scalar("ucb_min", np.min(gap), self.iteration)
        pulled_arm = np.argmax(ucb)
        self.arm_pulled_times[pulled_arm] += 1
        print("info", self.iteration)
        return pulled_arm