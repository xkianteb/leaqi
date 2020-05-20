import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch
from gym.envs.registration import registry as gym_registry

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Averaging():
    "Simple averaging."

    def __init__(self):
        self.count = 0.0
        self.value = 0.0

    def update(self, x):
        if x is None: return
        self.count += 1.0
        self.value += float(x)

    def __call__(self):
        if self.count == 0: return 0
        return self.value / self.count

    def reset(self):
        self.count = 0.0
        self.value = 0.0

class TrainLog:
    def __init__(self, dataset=None, train_y=None, weak_labeler=None, device='cuda',\
                 adjusted_classes=None, num_epoch=0, sparse=False, args=None):

        self.epoch = 0
        self.num_epoch = num_epoch
        self.args=args
        self.reset_epoch_counts()

    def reset_epoch_counts(self):
        # Epoch counts
        self.strongoracle_counts = 0
        self.weakoracle_counts = 0
        self.nocall_counts = 0
        self.d_classifier_counts = 0

    def update(self, strong=0, weak=0, nocall=0, dcupdate=0, classprob=None):
        if strong>0:
            self.strongoracle_counts += strong
        if weak>0:
            self.weakoracle_counts += weak
        if nocall>0:
            self.nocall_counts += nocall
        if dcupdate>0:
            self.d_classifier_counts += dcupdate

    def __call__(self, epoch=0, agent=None, diff_clf=None, loss=0):
        self.epoch = epoch
        str = ''
        str += 'Epoch [%d/%d] \n Loss: %.4f\n'\
               %(self.epoch, self.num_epoch, loss)
        str += ' No Calls: %.2f | Strong Calls: %.2f | Weak Calls: %.2f \n'\
               %(self.nocall_counts, self.strongoracle_counts, self.weakoracle_counts)
        str += ' D-Classifier Updates: %.2f \n'\
               %(self.d_classifier_counts)
        return str
