import multiprocessing as mp
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
import csv
import gzip
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from flashtext import KeywordProcessor
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import os

class Annotator:
    def __init__(self, train=None, test=None, tag2idx=None, args=None):
        self.train = train
        self.test = test
        self.tag2idx = tag2idx

    def reset(self, seed):
        pass

    def __call__(self, obs=None):
        [expert_tag, ref_tag] = obs
        idx = expert_tag.item()
        return (idx, None)

class Reference:
    def __init__(self, train=None, test=None, tag2idx=None, args=None):
        self.train = train
        self.test = test
        self.tag2idx = tag2idx

    def reset(self, seed):
        pass

    def __call__(self, obs=None):
        [expert_tag, ref_tag] = obs
        idx = ref_tag.item()
        return (idx, None)
