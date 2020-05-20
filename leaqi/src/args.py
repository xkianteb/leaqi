import torch
import random
import numpy as np
import argparse

RANDOM_SEED = random.randint(0,1e+5)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def wso_args(parser):
    # generic model parameters
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--b", type=float, default=.5)
    parser.add_argument("--seed", type=float, default=RANDOM_SEED)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--task", choices=['multi', 'ner', 'gym'], default='multi')
    parser.add_argument("--filename", type=str, default='tmp')
    parser.add_argument("--weak_feature", action='store_true', default=False)
    parser.add_argument("--env", choices=['Ner-v0', 'Keyphrase-v0','Pos-v0'])
    parser.add_argument("--no_apple_tasting", action='store_true', default=False)


    # weak and strong oracle parameters
    parser.add_argument("--method", choices=['mm', 'sm' 'entropy', 'lc'], default='mm',\
                        help="Discriminant classifier type")
    parser.add_argument("--diff_clf_lr", type=float, default=1e-3,\
                        help="First value lr and second value lr false negative instances\
                              if the discriment type is gradient")
    parser.add_argument("--diff_clf_type", choices=['gradient', 'adj_prob', 'entropy'],\
                        default='adj_prob')
    parser.add_argument("--diff_clf_th", type=float, default=.35)
    parser.add_argument("--diff_clf_fn_g", type=float, default=1e-2)
    parser.add_argument("--unbias_weight", action='store_true', default=False)
    parser.add_argument('--betadistro', choices=['1', 'query'], default='1')
    parser.add_argument("--ref_type", choices=['normal', 'random'], default='normal')

    # baseline parameters
    #parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--alg", choices=['dagger:strong', 'dagger:weak', 'leaqi'], default='leaqi')
    parser.add_argument("--query_strategy", choices=['active', 'passive', 'None', 'random'])
