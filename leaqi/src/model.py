import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader, TensorDataset
import random

class Model:
    def __init__(self, net, lr, method, device, args, optimization_steps=None, env=None):
        self.env=env

        self.net = net
        self.method = method
        self.device = device
        #TODO: Pass in as a parameter
        self.batch_size = 32
        try:
            self.b = args.b
        except:
            self.b = .01

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optim = torch.optim.Adam(self.net.parameters(), lr = lr)

        self.episode_states = []
        self.episode_actions = []
        self.aggregated_states = []
        self.aggregated_actions = []

    def __call__(self,x):
        x = x if isinstance(x, torch.Tensor) else torch.Tensor(x).to(self.device)
        state_value = self.net.forward(x.view(1,-1)).view(-1)
        sample_prob = self._make_query(state_value)
        return [state_value, sample_prob]

    # Compute score
    def _make_query(self, value):
        value = value.detach()
        value = torch.softmax(value.view(-1), dim=0).cpu().numpy().flatten()

        # least confident
        if self.method == 'lc':
            score = -np.max(value, axis=1)
        # smallest margin
        elif self.method == 'sm':
            if np.shape(value)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-value, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])
        # entropy
        elif self.method == 'entropy':
            score = np.sum(-dvalue * np.log(value), axis=1)
        # max margin
        elif self.method == 'mm':
            scores = np.sort(value)
            score = scores[-1] - scores[-2]

        # sampling prob
        sample_prob = self.b / (self.b + score)
        return sample_prob

    def reset(self):
        pass

    def save(self, state_value, y, batch_length=None):
        self.episode_states.append(state_value.view(1, -1).detach().cpu())
        y = torch.LongTensor([y]).to('cpu')
        self.episode_actions.append(y)

    def update(self):
        self.aggregated_states += self.episode_states
        self.aggregated_actions += self.episode_actions

        data = list(zip(self.aggregated_states, self.aggregated_actions))
        random.shuffle(data)

        states, actions = zip(*data)
        states = torch.cat(states).float()
        actions = torch.cat(actions).long()

        trdata = DataLoader(TensorDataset(states,actions), num_workers=4,\
            batch_size = self.batch_size)

        total_loss = 0
        for i, batch in enumerate(trdata):
            self.optim.zero_grad()
            state, action  = batch
            prediction = self.net.forward(state.to(self.device))
            loss = self.criterion(prediction, action.to(self.device))
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

        del self.episode_states[:]
        del self.episode_actions[:]

#    #TODO: Delete
#    def adjust_learning_rate(self, batch_size):
#        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#        #lr = float(self.lr) / float(batch_size)
#        #for param_group in self.optim.param_groups:
#        #    param_group['lr'] = lr
#        pass
