import sklearn.metrics as metrics
import gym
import torch
import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils import data
from leaqi.envs.gym_structured_prediction.envs.dataset import pad
from collections import Counter


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def batch_evaluation(agent, args, device='cuda', wrapper=None, seed=None, env=None):
    with torch.no_grad():
        diff_y_pred = []
        diff_y_true = []
        diff_y = []
        diff_w = []

        model_y_pred = []
        model_y_true = []

        th = args.diff_clf_th
        threshold = np.vectorize(lambda score: 1.0 if (1-th) < score else 0.0)

        flatten = lambda l: [item for sublist in l for item in sublist]

        batch_size = 250
        num_prev_actions = 1
        if args.weak_feature:
            num_prev_actions += env.action_space.n
            emb_tags = {}
            for key in env.idx2tag.keys():
                emb_tags[key] = env.dataset.feature_function(torch.cuda.LongTensor([[key]]))

        action_idx = 0
        batch_iter = data.DataLoader(dataset=env.test_dataset, batch_size=batch_size,\
                                       shuffle=False, collate_fn=pad)
        for (words, x, is_heads, tags, y, rlb_tags, rlby, seqlens) in batch_iter:
            num_items = len(is_heads)
            x = env.dataset.feature_function(x)
            is_heads = boolean_indexing(is_heads, fillval=0.0)
            is_heads[:,0] = 0 # masking out [CLS]
            seqlens = np.array(seqlens) - 1
            #is_heads[list(range(len(seqlens))), seqlens] = 0 # masking out [SEP]

            is_heads = [np.nonzero(is_heads[i])[0] for i in range(num_items)]
            # words = '[CLS] Nadim Ladki [SEP]'
            # words[0] = '[CLS]'
            index_of_heads = boolean_indexing(is_heads, fillval=0.0) # first index is 1
            state = torch.zeros(num_items,1, 768*(num_prev_actions+1)).to('cuda')

            batch_pred = [[] for _ in range(num_items)]
            batch_true = [[] for _ in range(num_items)]
            batch_w    = [[] for _ in range(num_items)]
            batch_diff = [[] for _ in range(num_items)]

            for i in range(index_of_heads.shape[1] - 1):
                embs = []
                ref_embs = []
                y_truth = []
                w_feature = []
                w_truth = []

                diff_states = []
                for batch_index in range(num_items):
                    start_index = int(index_of_heads[:,i][batch_index])
                    end_index = int(index_of_heads[:,i+1][batch_index])
                    embs.append(torch.mean(x[batch_index,start_index:end_index], dim=0).reshape(1,1,-1))
                    if args.weak_feature:
                        ref_embs.append(emb_tags[rlby[batch_index][start_index].item()])
                        w = rlby[batch_index][start_index].item()
                        w_feature.append(w)
                    # rlby[:,start_index:start_index+1]
                    y_truth.append(env.idx2tag[y[batch_index][start_index].item()])
                    if args.alg == 'leaqi':
                        w = rlby[batch_index][start_index].item()
                        w_truth.append(env.idx2tag[w])
                        diff_state = torch.zeros((1,1,agent.diff_input_size)).to(device)
                        diff_state[:,:,1536*w:1536*(w+1)] = state[batch_index]
                        diff_states.append(diff_state)


                embs = torch.cat(embs)
                state[:,:,768*num_prev_actions:] = embs
                state_value = agent.model.net.forward(state).squeeze()
                action = state_value.max(1)[1].view(num_items, -1)
                action_idx = 0 % num_prev_actions
                action_emb = env.dataset.feature_function(action)

                if args.alg == 'leaqi':
                    #diff_state = torch.zeros((1,1,9216)).to(self.device)
                    #diff_state[:,:,1536*y_ref:1536*(y_ref+1)] = state
                    diff_state = torch.cat(diff_states)
                    diff_state_value = agent.diff_clf.net(diff_state)
                    diff_score = torch.softmax(diff_state_value.squeeze(),dim=1)[:,1]
                    diff_score = diff_score.cpu().numpy().flatten()
                    diff_pred = threshold(diff_score).tolist()

                state[:,:,768*action_idx:768*(action_idx+1)] = action_emb

                action = [env.idx2tag[a] for a in action.view(-1).cpu().numpy().tolist()]
                for batch_index in range(len(action)):
                    if index_of_heads[:,i+1][batch_index]:
                        batch_pred[batch_index].append(action[batch_index])
                        batch_true[batch_index].append(y_truth[batch_index])
                        if args.alg == 'leaqi':
                            batch_w[batch_index].append(w_truth[batch_index])
                            batch_diff[batch_index].append(diff_pred[batch_index])

                        if args.weak_feature:
                            ref_idx = w_feature[batch_index]
                            #state[batch_index,:,768*(action_idx+1):768*(action_idx+2)] = ref_embs[batch_index]
                            state[batch_index,:,768*(action_idx+1+ref_idx):768*(action_idx+2+ref_idx)] = ref_embs[batch_index]


            model_y_pred += batch_pred
            model_y_true += batch_true
            if args.alg == 'leaqi':
                diff_true = (np.array(flatten(batch_w))==np.array(flatten(batch_true)))*1.0
                diff_y += np.array(flatten(batch_true)).tolist()
                diff_w += np.array(flatten(batch_w)).tolist()
                diff_y_true += diff_true.tolist()
                diff_y_pred += flatten(batch_diff)

    average_episodic_return = 0.0

    precision_macro, recall_macro, fscore_macro, _ =\
           metrics.precision_recall_fscore_support(flatten(model_y_true), flatten(model_y_pred), average='macro')
    precision_micro, recall_micro, fscore_micro, _ =\
           metrics.precision_recall_fscore_support(flatten(model_y_true), flatten(model_y_pred), average='micro')

    model_stats = {'acur': accuracy_score(model_y_true, model_y_pred),\
                   'f1': f1_score(model_y_true, model_y_pred),\
                   'precision': precision_score(model_y_true, model_y_pred),\
                   'recall': recall_score(model_y_true, model_y_pred),\
                   'avg_rtn': average_episodic_return,\
                   'sklearn_accuracy': metrics.accuracy_score(flatten(model_y_true), flatten(model_y_pred)),\
                   'sklearn_f1_macro': fscore_macro,\
                   'sklearn_precision_macro':  precision_macro,\
                   'sklearn_recall_macro': recall_macro, \
                   'sklearn_f1_micro': fscore_micro,\
                   'sklearn_precision_micro': precision_micro,\
                   'sklearn_recall_micro': recall_micro}

    # Tags level Metrics
    tags = list(env.tag2idx.keys())
    precision_all, recall_all, fscore_all, _ =\
           metrics.precision_recall_fscore_support(flatten(model_y_true), flatten(model_y_pred), average=None, labels=tags)

    for x in range(len(tags)):
        model_stats[f'{tags[x]}_f1_all'] = fscore_all[x]
        model_stats[f'{tags[x]}_recall_all'] = recall_all[x]
        model_stats[f'{tags[x]}_precision_all'] = precision_all[x]

    # O and not O Metrics
    precision_o_macro, recall_o_macro, fscore_o_macro, _ =\
           metrics.precision_recall_fscore_support(flatten(model_y_true), flatten(model_y_pred), average='macro', labels=['O'])
    model_stats['precision_o_macro'] = precision_o_macro
    model_stats['recall_o_macro'] = recall_o_macro
    model_stats['fscore_o_macro'] = fscore_o_macro

    if args.alg == 'leaqi':
        con_mat = metrics.confusion_matrix((1-np.array(diff_y_true)), (1-np.array(diff_y_pred))).ravel()
        eps = np.finfo(np.float32).eps.item()
        miss_rate = con_mat[2] / (con_mat[2] + con_mat[3] + eps)
        precision_diff, recall_diff, fscore_diff, _ =\
           metrics.precision_recall_fscore_support(np.array(diff_y_true), np.array(diff_y_pred), average='macro')

        diff_clf_stats = {'acur': metrics.accuracy_score(diff_y_true, diff_y_pred),
                          'con_mat': con_mat,\
                          'f1_macro':  metrics.f1_score(diff_y_true, diff_y_pred, average='macro'),
                          'f1_micro': metrics.f1_score(diff_y_true, diff_y_pred, average='micro'),
                          'miss_rate': miss_rate,
                          'precision_macro_diff': precision_diff,
                          'recall_macro_diff': recall_diff,
                          'fscore_macro_diff': fscore_diff}

        diff_true = np.array(diff_y_true)
        diff_pred = np.array(diff_y_pred)

        diff_true_count = Counter(diff_true)
        diff_pred_count = Counter(diff_pred)
        for key in diff_true_count.keys():
            diff_clf_stats[f'{key}_true_count'] = diff_true_count[key]

        for key in diff_pred_count.keys():
            diff_clf_stats[f'{key}_pred_count'] = diff_pred_count[key]

        for tag in set(diff_y):
            tag_idxs = np.argwhere(np.array(diff_y) == tag).flatten()
            tag_precision_diff, tag_recall_diff, tag_fscore_diff, _ =\
               metrics.precision_recall_fscore_support(diff_true[tag_idxs], diff_pred[tag_idxs], average='macro')
            diff_clf_stats[f'{tag}_f1_macro'] =tag_fscore_diff
            diff_clf_stats[f'{tag}_recall_macro'] = tag_recall_diff
            diff_clf_stats[f'{tag}_precision_macro'] = tag_precision_diff

            tag_precision_diff, tag_recall_diff, tag_fscore_diff, _ =\
               metrics.precision_recall_fscore_support(diff_true[tag_idxs], diff_pred[tag_idxs], average='micro')
            diff_clf_stats[f'{tag}_f1_micro'] =tag_fscore_diff
            diff_clf_stats[f'{tag}_recall_micro'] = tag_recall_diff
            diff_clf_stats[f'{tag}_precision_micro'] = tag_precision_diff
    else:
        diff_clf_stats = {'acur':0,\
                          'con_mat': (0,0,0,0),\
                          'f1_macro':  0,\
                          'f1_micro': 0,\
                          'miss_rate': 0,\
                          'precision_macro_diff': 0,\
                          'recall_macro_diff': 0,\
                          'fscore_macro_diff': 0}


    return (model_stats, diff_clf_stats)
