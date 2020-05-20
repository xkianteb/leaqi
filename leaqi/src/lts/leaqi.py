import numpy as np
import random
from numpy.random import binomial
from copy import copy, deepcopy
from collections import defaultdict

import torch
from leaqi.src.net import Net
from leaqi.src.model import Model
from leaqi.src.differenceclassifier import DifferenceClassifier

class LEAQI(object):
    def __init__(self, *, reference, expert, args,  device, env, trainlog):
        self.reference = reference
        self.expert = expert
        self.device = device
        self.args = args
        self.num_actions = env.action_space.n
        self.no_apple_tasting = args.no_apple_tasting
        self.if_query = [False for _ in range(env.action_space.n)]
        self.if_mistake = [False for _ in range(env.action_space.n)]
        self.mistakes = [0 for _ in range(env.action_space.n)]
        self.queries = [1 for _ in range(env.action_space.n)]

        # Create the learner model
        try:
            input_size = env.reset().size
            assert(isinstance(input_size, int))
        except:
            input_size = env.reset().cpu().numpy().size

        # Create the learner model
        model_net = Net(in_features=input_size, out_features=env.action_space.n).to(device)
        self.model = Model(model_net, args.model_lr, args.method, device, args)

        # Create the difference classifier
        self.diff_input_size=input_size*env.action_space.n
        args.num_actions = env.action_space.n
        dc_net = Net(in_features=self.diff_input_size, out_features=2).to(device)
        self.diff_clf = DifferenceClassifier(dc_net, args.diff_clf_type,\
                                           args.diff_clf_lr, device, args.diff_clf_th,\
                                           args=args)
        self.trainlog = trainlog
        self.unseen_mistakes = [0 for _ in range(env.action_space.n)]
        self.other_mistakes = [0 for _ in range(env.action_space.n)]
        self.seen_mistakes = [0 for _ in range(env.action_space.n)]
        self.total_mistakes = [0 for _ in range(env.action_space.n)]
        self.random_queries = [0 for _ in range(env.action_space.n)]
        self.model_mistakes = [0 for _ in range(env.action_space.n)]

    def obs(self, env):
        obs = (env.expert_tag, env.ref_tag)
        return obs

    def reset(self, seed=0):
        self.expert.reset(seed)
        self.reference.reset(seed)
        self.model.optim.zero_grad()
        self.diff_clf.optim.zero_grad()

    def appletaste(self, d, y_ref):
        query_weights = [self.queries[x]/sum(self.queries) for x in range(len(self.queries))]
        alpha = 1-query_weights[y_ref]
        beta = 1 if self.args.betadistro == '1' else (1+query_weights[y_ref])
        if d == 0:
            query = 0
        elif ( d == 1 and np.random.beta(alpha, beta) <= ((self.mistakes[y_ref] + 1)/self.queries[y_ref])**(1./2.)):
            query = 0
            self.random_queries[y_ref] += 1
        elif d == 1:
            query = 1
        return query

    def __call__(self, env, state):
        # receive instance and weak label
        [y_ref, top_choices] = self.reference(obs=self.obs(env))
        [y_expert, _] = self.expert(obs=self.obs(env))

        if self.args.ref_type == 'random':
            y_ref = env.action_space.sample()

        with torch.no_grad():
            [state_value, sample_prob] = self.model(state)
        y_hat = state_value.max(0)[1].item() # max returns (value, index)

        if binomial(1, sample_prob):
            with torch.no_grad():
                diff_state = torch.zeros((1,1,self.diff_input_size)).to(self.device)
                diff_state[:,:,1536*y_ref:1536*(y_ref+1)] = state
                [d, diff_state_value] = self.diff_clf(diff_state)

            if self.no_apple_tasting:
                query = d
            else:
                query = self.appletaste(d,y_ref)

            if query:
                # AGREEMENT
                # Query weak oracle
                y_star = y_ref
                self.trainlog.update(weak=1)
                if int(y_ref==y_expert) == 0:
                    self.unseen_mistakes[y_expert] += 1
                self.queries[y_ref] += 1
            else:
                self.queries[y_expert] += 1
                # NO AGREEMENT
                y_star = y_expert

                # Update the log
                self.trainlog.update(strong=1)

                # Update Difference classifier
                self.trainlog.update(dcupdate=1)
                self.diff_clf.save(diff_state, y_ref==y_expert, type=y_expert)

                if d == 1 and query == 0 and int(y_ref==y_expert) == 0:
                    self.mistakes[y_ref] += 1

                if d == 1 and int(y_ref==y_expert) == 0:
                    self.seen_mistakes[y_expert] += 1

            # Accumulate gradients
            self.model.save(state, y_star)
        else:
            self.trainlog.update(nocall=1)
            if int(y_hat==y_expert) == 0:
                self.model_mistakes[y_expert] += 1
                self.total_mistakes[y_expert] += 1
        return y_hat

    def update(self):
        self.model.update()
        self.diff_clf.update()
