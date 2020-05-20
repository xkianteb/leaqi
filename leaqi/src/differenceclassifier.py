import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import random
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
from leaqi.src.net import Net

class DifferenceClassifier:
    def __init__(self, net, dc_type, lr, device, th, optimization_steps=None, args=None):
        self.net = net
        self.dc_type = dc_type
        self.device = device
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_size = 32
        self.unbias_weight = args.unbias_weight

        self.episode_states = []
        self.episode_truths = []
        self.episode_weights = []
        self.episode_weights_dict = defaultdict(list)

        self.aggregated_states = []
        self.aggregated_truths = []
        self.aggregated_weights = []
        self.aggregated_weights_dict =  defaultdict(list)

        # Adjusted probabilty changes the prob so that it is really sensitive to
        # false negatives (i.e. saying that they agree but they dont)
        # Lower reduces false negatives and recall
        self.pred = lambda x: self._pred_threshold(x, th=th)

        if isinstance(lr, str):
            lr = lr.split()

        self.lr = float(lr)
        #self.optim = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        if self.unbias_weight:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss()

    def __call__(self, x):
        x = x if isinstance(x, torch.Tensor) else torch.Tensor(x).to(self.device)
        return self.pred(self.net(x.view(1,-1)))

    def _pred_threshold(self, y_score, th=None):
        # 1.0 means agree --> query weak
        # 0.0 means disagree --> query strong
        y_score = y_score if len(y_score.shape) == 1 else y_score.view(-1)
        score = torch.softmax(y_score.detach(), dim=0).cpu().numpy().flatten()
        assert(len(score) == 2)
        score = score[1]
        agree = 1.0 if (1-th) < score else 0.0
        return agree, y_score.view(-1)

    def save(self, state, truth, type):
        self.episode_states.append( state.view(1,-1).detach().cpu() )
        self.episode_truths.append( torch.LongTensor([truth]) )
        self.episode_weights.append(type)
        self.episode_weights_dict[type].append(truth*1.0)

    def update(self):
        self.aggregated_states += self.episode_states
        self.aggregated_truths += self.episode_truths
        self.aggregated_weights += self.episode_weights
        for key in self.episode_weights_dict.keys():
            self.aggregated_weights_dict[key] += self.episode_weights_dict[key]

        if self.unbias_weight:
            weights_counts = {}
            for key in self.aggregated_weights_dict.keys():
                keys = np.sort(np.unique(self.aggregated_weights_dict[key]))
                class_weight = compute_class_weight('balanced', keys, self.aggregated_weights_dict[key])
                class_weight = defaultdict(int, dict(zip(keys, class_weight)))
                weights_counts[key] = {0.:class_weight[0.], 1.:class_weight[1.]}

        data = list(zip(self.aggregated_states, self.aggregated_truths, self.aggregated_weights))
        random.shuffle(data)

        states, truths, weights = zip(*data)
        states = torch.cat(states).float()
        truths = torch.cat(truths).long()
        weights = np.array(weights)

        trdata = DataLoader(TensorDataset(states,truths), num_workers=4,\
            batch_size = self.batch_size)

        total_loss = 0
        for i, batch in enumerate(trdata):
            self.optim.zero_grad()
            state, truth  = batch

            prediction = self.net.forward(state.to(self.device))
            loss = self.criterion(prediction, truth.to(self.device))
            # Scale the loss protion to class size
            if self.unbias_weight:
                start_w = i * self.batch_size
                end_w = start_w + state.shape[0]
                weights_vec = []

                for idx in range(start_w, end_w):
                    weights_vec.append(weights_counts[weights[idx]][truths[idx].item()])
                weights_vec = torch.from_numpy(np.array(weights_vec)).to(self.device)
                loss *= weights_vec
                loss = loss.mean()
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

        del self.episode_states[:]
        del self.episode_truths[:]
        del self.episode_weights[:]
        self.episode_weights_dict = defaultdict(list)

    #def adjust_learning_rate(self, batch_size):
    #    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #    #lr = float(self.lr) / float(batch_size)
    #    #for param_group in self.optim.param_groups:
    #    #    param_group['lr'] = lr
    #    pass
