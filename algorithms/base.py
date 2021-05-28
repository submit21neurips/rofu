import numpy as np
import torch
import torch.nn.functional as F
from algorithms.models.utils import build_model

class Base:
    def __init__(self,
                 writer,
                 model_config,
                 batch_size=-1,
                 learning_rate=0.01,
                 train_lr_decay=0.,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                ):
        self.writer = writer
        self.batch_size = batch_size
        self.train_data_epoch = train_data_epoch
        self.shared_parameter = shared_parameter
        self.loss_type = loss_type
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.context_shape = (3, 32, 32) if model_config['type'][:-2] == 'resnet' else (model_config['input_size'],) 
        self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        fcn = lambda step: 1./(1. + train_lr_decay*step)

        def fcn(step):
            _lr_coef = 1. / (1. + train_lr_decay * step)
            return max(0.05, _lr_coef)


        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fcn)
        self.loss_func = torch.nn.MSELoss() if loss_type == 'mse' else lambda pred, train: -torch.mean(pred * torch.log(train) + (1 - pred) * torch.log(1 - train))
        
        self.reset()
        
    def reset(self):
        self.iteration = 0
        self.actions_list = []
        self.rewards_list = []
        self.contexts_list = []
        self.actions = np.array([])
        self.rewards = np.array([])
        self.contexts = np.array([])
        
    def _build_model(self):
        self.model = build_model(self.model_config, self.device)
        
    def _sample_data(self):
        # sample data from buffer
        if self.batch_size == -1:
            iterations_so_far = np.arange(self.rewards.shape[0])
        else:
            iterations_so_far = np.random.choice(range(self.rewards.shape[0]), self.batch_size, replace=True)
        x_train = torch.FloatTensor(self.contexts[iterations_so_far]).to(self.device)
        y_train = torch.FloatTensor(self.rewards[iterations_so_far]).to(self.device)
        actions_so_far = self.actions[iterations_so_far]
        return x_train, y_train , actions_so_far
    
    def _reset_opt_lr(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate / (1 + 0.01 * self.iteration)
            
    def _train_on_data(self):
        if self.actions.size == 0:
            return
        self.model.train()
        current_loss = 0
        for epoch in range(self.train_data_epoch):
            x_train, y_train, actions_so_far = self._sample_data()
            if self.shared_parameter:
                raise NotImplemented
                y_pred = self.model.forward(x_train).squeeze()
            else:
                if self.loss_type == 'mse':
                    y_pred = self.model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far].squeeze()
                else:
                    y_pred = torch.exp(self.model.forward(x_train)[np.arange(actions_so_far.shape[0]), actions_so_far]).squeeze()
            # print("y_pred", y_pred)                    
            # print("y_train", y_train)

            loss = self.loss_func(y_pred, y_train)
            current_loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
        self.lr_scheduler.step()
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        current_lr = get_lr(self.optimizer)
        #self.writer.add_scalar("current_lr", current_lr, self.iteration)
        self.writer.add_scalar("training loss", loss, self.iteration)
        return current_loss
            
        
    def update(self, action, context, reward):
        self.actions_list.append(action)
        assert self.shared_parameter == False
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
        raise NotImplemented('Should implement how to make decisions.')