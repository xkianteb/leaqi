import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils import data

import pandas as pd
from sklearn.model_selection import train_test_split
import scipy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NewSentence(Exception):
    pass

class Dataset(object):
    def __init__(self, train_dataset=None, test_dataset=None, num_classes=None,\
                       features=None, idx2tag=None, tag2idx=None):
        self.num_classes = num_classes if num_classes is not None\
                                else np.unique(train_y).shape[0]

        self.train_X = np.array(train_dataset.org_sents)
        self.train_y = np.array(train_dataset.org_tags)
        self.train_rlby = np.array(train_dataset.org_rlb_tags)

        self.test_X = np.array(test_dataset.org_sents)
        self.test_y = np.array(test_dataset.org_tags)
        self.test_rlby = np.array(test_dataset.org_rlb_tags)

        self.num_train_samples = self.train_X.shape[0]
        self.num_test_samples = self.test_X.shape[0]

        self.feature_function = features
        self.word_mask = torch.zeros((1,200)).long()

        self.idx2tag = idx2tag
        self.tag2idx = tag2idx

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=1,\
                               shuffle=True, collate_fn=pad, num_workers=4)

        self.test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=500,\
                              shuffle=False, collate_fn=pad, num_workers=4)

        self.train_iter = iter(self.train_dataloader)
        self.test_iter = iter(self.test_dataloader)

    def reset(self):
        self.train_iter = iter(self.train_dataloader)
        self.test_iter = iter(self.test_dataloader)

    def get_input_data(self, data_type=None):
        if data_type == 'train':
            return self.train_X
        elif data_type == 'test':
            return self.test_X
        else:
            raise Exception('Unknow data type')

    def __call__(self, data_type = None):
        self.data_type = data_type
        try:
            if data_type == 'train':
                [words, x, is_heads, tags, y, rlb_tags, rlby, seqlen] = next(self.train_iter)
            elif data_type =='test':
                [words, x, is_heads, tags, y, rlb_tags, rlby, seqlen] = next(self.test_iter)
            else:
                raise Exception('Unknow data type')

            x = self.feature_function(x)
            is_heads = is_heads[0]
            is_heads[0] = 0 # masking out [CLS]
            is_heads[-1] = 0 # masking out [SEP]
            index_of_heads = np.nonzero(is_heads)[0].tolist() + [seqlen[0] - 1]
            for i in range(len(index_of_heads[:-1])):
                start_index = index_of_heads[i]
                end_index = index_of_heads[i+1]
                # Bert tokenizer does WordPiece
                # example: SOCCER -> ['S', '##OC', '##CE', '##R']
                # below averages embeddings of the pieces

                # [word, self.expert_tag, self.ref_tag, self.previous_tag]
                word = torch.mean(x[:,start_index:end_index], dim=1).reshape(1,1,-1)
                expert_tag = y[:,start_index:start_index+1]
                ref_tag = rlby[:,start_index:start_index+1]
                yield(word, expert_tag, ref_tag)
            raise NewSentence
        except StopIteration:
            raise StopIteration

def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    rlb_tags = f(5)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    y = f(-4, maxlen)
    rlby = f(-2, maxlen)

    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), rlb_tags, f(rlby), seqlens
