import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import warnings
import uuid
import random
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # NOTE: pytorch_pretrained_bert and transformers tokenizers are different
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    from transformers import AutoModel, AutoTokenizer
from gym import error, spaces
from leaqi.envs.gym_structured_prediction.envs.dataset import Dataset, NewSentence
import numpy as np
import gym

class StructuredPredictionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, *, num_prev_actions, update_interval, bert_model, VOCAB, ID):
        self.tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

        dir = os.path.dirname(os.path.abspath(__file__))

        if ID.lower() == 'pos':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
            train_filename = os.path.join(dir,'pos_dataset_files', 'train_UD_Greek-GDT_pos.txt')
            test_filename = os.path.join(dir, 'pos_dataset_files', 'test_UD_Greek-GDT_pos.txt')
        elif ID.lower() == 'ner':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
            train_filename = os.path.join(dir,'ner_dataset_files', 'train_gazetter.txt')
            test_filename = os.path.join(dir, 'ner_dataset_files', 'test_gazetter.txt')
        elif ID.lower() == 'keyphrase':
            #Download and save the scibert model and tokenizer
            num = random.randint(0, 10000)
            envs_folder = os.path.dirname(os.path.abspath(__file__))
            scibert_folder = os.path.join(envs_folder,f"scibert_files_{num}")
            os.system(f'mkdir -p {scibert_folder}')
            if not os.path.exists(os.path.join(scibert_folder, 'tokenizer_config.json')):
                tokenizer =  AutoTokenizer.from_pretrained(f'allenai/{bert_model}')
                tokenizer.save_pretrained(scibert_folder)
            if not os.path.exists(os.path.join(scibert_folder, 'pytorch_model.bin')):
                model = AutoModel.from_pretrained(f'allenai/{bert_model}')
                model.save_pretrained(scibert_folder)

            #self.tokenizer = AutoTokenizer.from_pretrained(f'allenai/{bert_model}', do_lower_case=False)
            self.tokenizer = BertTokenizer.from_pretrained(scibert_folder, do_lower_case=False)
            train_filename = os.path.join(dir, 'keyphrase_dataset_files', 'train_semaeval_keyphrase.txt')
            test_filename = os.path.join(dir, 'keyphrase_dataset_files', 'test_semaeval_keyphrase.txt')
            bert_model=scibert_folder
        else:
            raise Exception(f'Unknown ID type: {ID}')

        self.train_dataset = train_dataset = DatasetLoader(train_filename, tokenizer=self.tokenizer, tag2idx=self.tag2idx)
        self.test_dataset =  test_dataset = DatasetLoader(test_filename, tokenizer=self.tokenizer, tag2idx=self.tag2idx)

        features = BertFeatures(vocab_size=len(VOCAB), device='cuda:0', bert_model=bert_model)
        #features.to('cuda:0')
        features = features.cuda()
        features = nn.DataParallel(features)

        self.num_prev_actions = num_prev_actions
        self.dataset = Dataset(train_dataset=train_dataset, test_dataset=test_dataset,\
                               num_classes=len(VOCAB), features=features,\
                               tag2idx=self.tag2idx, idx2tag=self.idx2tag)
        self.sent_iter = self.dataset()
        self.state = torch.zeros(1,1, 768*(num_prev_actions+1)).to('cuda:0')

        self.expert_tag = None
        self.ref_tag = None

        self.action_idx = 0 # keeps track of action history
        self.sent_idx = 0 # keeps track of sentences
        self.word_idx = 0 # keeps track of word indices in sentences
        self.update_interval = update_interval
        self.update = False
        self.sent_idx_length = len(self.train_dataset.sents[self.sent_idx]) - 2

        self.action_space = spaces.Discrete(len(VOCAB))
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(768*num_prev_actions+1,))
        self.weak_feature = None

        self.vocab_emb = {}

    def step(self, action):
        reward = (self.expert_tag.item() == action) * 1.0
        info = {'stopiteration': False}
        done = False

        try:
            [word, self.expert_tag, self.ref_tag] = next(self.sent_iter)
            self.state[:,:,768*self.num_prev_actions:] = word

            self.word_idx += 1
            self.update = False

            # Update next state with previous action
            self.action_idx = 0 % self.num_prev_actions
            action_emb = self.dataset.feature_function(torch.cuda.LongTensor([[action]]))
            self.state[:,:,768*self.action_idx:768*(self.action_idx+1)] = action_emb
            if self.weak_feature is not None:
                ref_idx = self.ref_tag.item()
                ref_emb = self.dataset.feature_function(torch.cuda.LongTensor([[self.ref_tag]]))
                self.state[:,:,768*(self.action_idx+1+ref_idx):768*(self.action_idx+2+ref_idx)] = ref_emb
            self.action_idx += 1

            state = self.state.clone()
        except NewSentence:
            self.update = True
            try:
                self.state.fill_(0)
                self.action_idx = 0
                self.sent_iter = self.dataset(self.data_type)
                [word, self.expert_tag, self.ref_tag] = next(self.sent_iter)
                self.state[:,:,768*self.num_prev_actions:] = word
                state = self.state.clone()

                self.sent_idx += 1
                self.word_idx = 0
                self.sent_idx_length = len(self.train_dataset.sents[self.sent_idx]) - 2
            except StopIteration:
                done = True
                state = None
                info['stopiteration'] = True

                self.sent_idx = 0
                self.word_idx = 0

        done = True if done or self.update else False
        return state, reward, done, info

    def seed(self, seed):
        pass

    def set_state(self, args=None):
        self.weak_feature = args.weak_feature
        if self.weak_feature:
            #TODO: Fix Hack, increasing num_prevous by 1 to take into account
            # ref tag feature
            self.num_prev_actions += self.action_space.n
            self.state = torch.zeros(1,1, 768*(self.num_prev_actions+1)).to('cuda:0')
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(768*self.num_prev_actions+1,))

    def reset(self, data_type='train'):
        self.data_type=data_type
        if self.sent_idx == 0:
            self.dataset.reset()
            self.sent_iter = self.dataset(data_type=data_type)

            self.state.fill_(0)
            self.action_idx = 0
            [word, self.expert_tag, self.ref_tag] = next(self.sent_iter)
            if self.weak_feature is not None:
                ref_emb = self.dataset.feature_function(torch.cuda.LongTensor([[self.ref_tag]]))
                ref_idx = self.ref_tag.item()
                self.state[:,:,768*(self.action_idx+1+ref_idx):768*(self.action_idx+2+ref_idx)] = ref_emb

            self.sent_idx = 0
            self.word_idx = 0

        state = self.state.clone()
        return state

class DatasetLoader:
    def __init__(self, fpath, shuffle=False, tokenizer=None, tag2idx=None):
        """
        fpath: [train|valid|test].txt
        """
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li,rlb_tags_li = [], [], [] # list of lists
        org_sents, org_tags, org_rlb_tags = [], [], []

        for entry in entries:
            if entry.startswith('-DOCSTART'): continue
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            rlb_tags = ([line.split()[-2] for line in entry.splitlines()])

            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
            rlb_tags_li.append(["<PAD>"] + rlb_tags + ["<PAD>"])

            org_sents.append(words)
            org_tags.append(tags)
            org_rlb_tags.append(rlb_tags)

        self.sents, self.tags_li, self.rlb_tags_li = sents, tags_li, rlb_tags_li
        self.org_sents, self.org_tags, self.org_rlb_tags = org_sents, org_tags, org_rlb_tags

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        # words, tags: string list
        words = self.sents[idx]
        tags = self.tags_li[idx]
        rlb_tags = self.rlb_tags_li[idx]

        # We give credits only to the first piece.
        x, y, rlby = [], [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t, rlbt in zip(words, tags, rlb_tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2idx[each] for each in t]  # (T,)

            rlbt = [rlbt] + ["<PAD>"] * (len(tokens) - 1)
            rlbyy = [self.tag2idx[each] for each in rlbt]

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            rlby.extend(rlbyy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        rlb_tags = " ".join(rlb_tags)
        return words, x, is_heads, tags, y, rlb_tags, rlby, seqlen

class BertFeatures(nn.Module):
    def __init__(self, vocab_size=None, device='cuda:0', bert_model=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.device = device

    def forward(self, x):
        '''
        x: (N, T). int64
        y: (N, T). int64
        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        self.bert.eval()
        #with torch.no_grad():
        #    outputs = self.bert(x)
        #    last_hidden_states = outputs[0]
        #return last_hidden_states
        with torch.no_grad():
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        return enc
