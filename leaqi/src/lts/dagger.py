import argparse
import random
from numpy.random import binomial

import gym
import numpy as np
import torch

from leaqi.src.model import Model
from leaqi.src.net import Net
from leaqi.src.differenceclassifier import DifferenceClassifier

class DAgger(object):
    def __init__(self, oracle_policy=None, device=None, env=None, sparse=None, trainlog=None, args=None):
        assert(args.query_strategy in ['active', 'passive', 'random'])

        self.expert = oracle_policy
        self.query_strategy = args.query_strategy
        self.device = device
        self.args = args
        self.batch_size = 0

        self.y_1_data = 0
        self.y_0_data = 0

        # Create the learner model
        try:
            input_size = env.reset().size
            assert(isinstance(input_size, int))
        except:
            input_size = env.reset().cpu().numpy().size

        model_net = Net(in_features=input_size, out_features=env.action_space.n).to(device)
        self.model = Model(model_net, args.model_lr, args.method,device, args, env=env)

        self.trainlog = trainlog

        ## TODO: Fix Trainlog and remove this
        dc_net = Net(in_features=input_size, out_features=2).to(device)
        self.diff_clf = DifferenceClassifier(dc_net, args.diff_clf_type,\
                                     args.diff_clf_lr, device,  args.diff_clf_th,\
                                     args=args)

        self.unseen_mistakes = [0 for _ in range(env.action_space.n)]
        self.other_mistakes = [0 for _ in range(env.action_space.n)]
        self.seen_mistakes = [0 for _ in range(env.action_space.n)]
        self.total_mistakes = [0 for _ in range(env.action_space.n)]
        self.random_queries = [0 for _ in range(env.action_space.n)]
        self.model_mistakes = [0 for _ in range(env.action_space.n)]
        self.mistakes = [0 for _ in range(env.action_space.n)]
        self.queries = [0 for _ in range(env.action_space.n)]
        self.if_query = [False for _ in range(env.action_space.n)]
        self.num_actions = env.action_space.n

    def reset(self, seed=0):
        self.expert.reset(seed)
        self.model.optim.zero_grad()
        self.batch_size = 0
        self.y_1_data = 0
        self.y_0_data = 0

    def obs(self, env):
        obs = (env.expert_tag, env.ref_tag)
        return obs

    def __call__(self, env, state):
        with torch.no_grad():
            [state_value, sample_prob] = self.model(state)
        y_hat = state_value.max(0)[1].item() # max returns (value, index)
        [y_expert, _] = self.expert(obs=self.obs(env))

        if (self.query_strategy == 'active' and binomial(1, sample_prob)) or\
           (self.query_strategy == 'passive') or\
           (self.query_strategy == 'random' and binomial(1, .5)):
            # Query the reference policy
            y_star = y_expert

            # Update the log
            if 'strong' in self.args.alg:
                self.trainlog.update(strong=1)
            elif 'weak' in self.args.alg:
                self.trainlog.update(weak=1)

            # Accumulate gradients
            batch_length = env.sent_idx_length
            self.model.save(state, y_star)
            self.batch_size +=1
        elif self.query_strategy == 'active':
            if 'strong' in self.args.alg or self.query_strategy == 'random':
                self.trainlog.update(nocall=1)

            if int(y_hat==y_expert) == 0:
                self.model_mistakes[y_expert] += 1
                self.total_mistakes[y_expert] += 1
        return y_hat

    def update(self):
        self.model.update()
