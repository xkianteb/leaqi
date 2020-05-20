import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import gym
from timeit import default_timer as timer
from datetime import timedelta
import torch
import argparse
import numpy as np
import time
from leaqi.src.util import TrainLog
from leaqi.src.args import wso_args
from leaqi.src.lts.leaqi import LEAQI
from leaqi.src.lts.dagger import DAgger
from leaqi.src.oracles import Reference, Annotator
from leaqi.envs import gym_structured_prediction
from leaqi.src.batch_evaluation import batch_evaluation

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main(args, _print=False):
    env = None
    env = gym.make(args.env)
    if args.weak_feature:
        env.set_state(args=args)
    device = 'cuda'
    wrapper = None

    #TODO: Overwriting this method to meean reference
    ref = Reference(train=env.dataset.train_rlby, test=env.dataset.test_rlby,\
                      tag2idx=env.dataset.tag2idx, args=args)
    expert = Annotator(train=env.dataset.train_y, test=env.dataset.test_y,\
                        tag2idx=env.dataset.tag2idx, args=args)
    args.num_epochs = 100000000000
    eval = 1000

    trainlog = TrainLog(args=args, num_epoch=args.num_epochs)

    if args.alg == 'leaqi':
        agent = LEAQI(reference=ref, expert=expert, args=args, device=device,\
                      env=env, trainlog=trainlog)
        filename = f'{args.alg}'
    elif args.alg == 'dagger:strong':
        agent = DAgger(oracle_policy=expert, args=args, env=env,\
                       trainlog=trainlog, device=device)
        filename = f'{args.alg}_{args.query_strategy}'
    elif args.alg == 'dagger:weak':
        agent = DAgger(oracle_policy=ref, args=args, env=env,\
                       trainlog=trainlog, device=device)
        filename = f'{args.alg}_{args.query_strategy}'
    else:
        raise Exception("Unknown algoritm")

    num_steps = 0
    num_fn = 0
    data = []
    try:
        nextupdatestart = timer()
        for traj in range(args.num_epochs):
            agent.reset(args.seed)
            state = env.reset()
            env.seed(args.seed)
            ep_reward = 0

            trainstart = timer()
            allstart = timer()
            while True:
                num_steps +=1
                action = agent(env, state)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                if done:
                    break
            agent.update()
            trainend = timer()

            if traj % eval == 0:
                [model_stats, diff_clf_stats] =\
                        batch_evaluation(agent, args, wrapper=wrapper, seed=args.seed, env= env)
                allend = timer()
                nextupdateend = timer()

                # Print Results
                stats = trainlog(traj, agent=agent, diff_clf=agent.diff_clf, loss=0)
                stats += f' traj: {traj}\n'
                stats += f' num_steps: {num_steps}\n'
                if args.alg == 'leaqi':
                    stats += f' ****************\n'
                    stats += f' Difference Classifier:\n'
                    stats += f'     Acuracy  : {diff_clf_stats["acur"]}\n'
                    stats += f'     F1 macro : {diff_clf_stats["f1_macro"]}\n'
                    stats += f'     Miss rate: {diff_clf_stats["miss_rate"]}\n'
                    stats += f'     (tn, fp, fn, tp): {tuple(diff_clf_stats["con_mat"])}\n'
                stats += f' *****************\n'
                stats += f' Model Classifier:\n'
                stats += f'     Acuracy  : {model_stats["acur"]}\n'
                stats += f'     F1       : {model_stats["f1"]}\n'
                stats += f'     sklearnf1: {model_stats["sklearn_f1_macro"]}\n'
                stats += f'     Train Reward: {ep_reward}\n'
                stats += f'     Test Reward:  {model_stats["avg_rtn"]}\n'
                stats += f' *****************\n'
                stats += f' Model Mistakes\n'
                stats += f'     ref queries: {trainlog.weakoracle_counts}\n'
                stats += f'     exp queries: {trainlog.strongoracle_counts}\n'
                stats += f'     no queries: {trainlog.nocall_counts}\n'
                stats += f'     ########################################\n'
                stats += f'     queries: {agent.queries}\n'
                stats += f'     unseen_mistakes : {agent.unseen_mistakes}\n'
                stats += f'     model_mistakes: {agent.model_mistakes}\n'
                stats += f'     seen mistakes: {agent.seen_mistakes}\n'
                stats += f'     mistakes: {agent.mistakes}\n'
                stats += f'     total: {agent.total_mistakes}\n'
                stats += f' Train/Update: {timedelta(seconds=trainend-trainstart)}\n'
                stats += f' Train/Update/Evaluate: {timedelta(seconds=allend-allstart)}\n'
                stats += f' Between Update: {timedelta(seconds=nextupdateend-nextupdatestart)}\n'
                stats += f'-----------------------------------'
                print(stats)

                # Log Results
                results ={'x': num_steps, 'train_accuracy':ep_reward,\
                     'test_accuracy':reward,\
                     'expert_queries':trainlog.strongoracle_counts,\
                     'ref_queries': trainlog.weakoracle_counts,\
                     'no_queries': trainlog.nocall_counts,\
                     'diff_clf_acur': diff_clf_stats["acur"],\
                     'tn': diff_clf_stats["con_mat"][0], 'fp': diff_clf_stats["con_mat"][1],\
                     'fn': diff_clf_stats["con_mat"][2], 'tp': diff_clf_stats["con_mat"][3],\
                     'diff_clf_f1':diff_clf_stats['f1_macro']}

                for idx in range(len(agent.random_queries)):
                     results[f'mistake_{env.idx2tag[idx]}_agent_random_queries']= agent.random_queries[idx]
                     results[f'mistake_{env.idx2tag[idx]}_agent_seen_mistakes']=  agent.seen_mistakes[idx]
                     results[f'mistake_{env.idx2tag[idx]}_agent_total_mistakes']= agent.total_mistakes[idx]
                     results[f'mistake_{env.idx2tag[idx]}_agent_unseen_mistakes']=agent.unseen_mistakes[idx]
                     results[f'mistake_{env.idx2tag[idx]}_agent_other_mistakes']= agent.other_mistakes[idx]
                     results[f'mistake_{env.idx2tag[idx]}_queries']= agent.queries[idx]


                for key in model_stats.keys():
                    results[f'model_{key}'] = model_stats[key]

                for key in diff_clf_stats.keys():
                    results[f'diff_{key}'] = diff_clf_stats[key]

                data.append(results)
                filename = args.filename if args.filename != 'tmp' else filename
                pd.DataFrame(data).to_csv(f'{filename}_results.csv')
                nextupdatestart = timer()
            else:
                allend = timer()
                stats =  f' traj: {traj}\n'
                stats += f' num_steps: {num_steps}\n'
                stats += f' *****************\n'
                stats += f' Model Mistakes\n'
                stats += f'     ref queries: {trainlog.weakoracle_counts}\n'
                stats += f'     exp queries: {trainlog.strongoracle_counts}\n'
                stats += f'     no queries: {trainlog.nocall_counts}\n'
                stats += f'     ########################################\n'
                stats += f'     queries: {agent.queries}\n'
                stats += f'     unseen_mistakes : {agent.unseen_mistakes}\n'
                stats += f'     model_mistakes: {agent.model_mistakes}\n'
                stats += f'     seen mistakes: {agent.seen_mistakes}\n'
                stats += f'     mistakes: {agent.mistakes}\n'
                stats += f'     total: {agent.total_mistakes}\n'
                stats += f' Train/Update: {timedelta(seconds=trainend-trainstart)}\n'
                stats += f' Train/Update/Evaluate: {timedelta(seconds=allend-allstart)}\n'
                stats += f'-----------------------------------'
                print(stats)


            if 'stopiteration' in info:
                if info['stopiteration']:
                    break
    except KeyboardInterrupt:
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wso_args(parser)
    args = parser.parse_args()
    print(args.filename)
    main(args)

#    trainlog = TrainLog(args=args, num_epoch=args.num_epochs)
#
#    if args.alg == 'leaqi':
#        agent = LEAQI(reference=ref, expert=expert, args=args, device=device,\
#                      env=env, trainlog=trainlog)
#        filename = f'{args.alg}_{args.no_apple_tasting}'
#    elif args.alg == 'dagger:strong':
#        agent = DAgger(oracle_policy=expert, args=args, env=env,\
#                       trainlog=trainlog, device=device)
#        filename = f'{args.alg}_{args.query_strategy}'
#    elif args.alg == 'dagger:weak':
#        agent = DAgger(oracle_policy=ref, args=args, env=env,\
#                       trainlog=trainlog, device=device)
#        filename = f'{args.alg}_{args.query_strategy}'
#    else:
#        raise Exception("Unknown algoritm")
#
#    num_steps = 0
#    num_fn = 0
#    data = []
#    try:
#        nextupdatestart = timer()
#        for traj in range(args.num_epochs):
#            agent.reset(args.seed)
#            state = env.reset()
#            env.seed(args.seed)
#            ep_reward = 0
#
#            trainstart = timer()
#            allstart = timer()
#            while True:
#                num_steps +=1
#                info = (num_fn, traj)
#                action = agent(env, state, info)
#                state, reward, done, info = env.step(action)
#                ep_reward += reward
#                if done:
#                    break
#            agent.update()
#            trainend = timer()
#
#            if traj % eval == 0:
#                [model_stats, diff_clf_stats] =\
#                        batch_evaluation(agent, args, wrapper=wrapper, seed=args.seed, env= env)
#                allend = timer()
#                nextupdateend = timer()
#
#                # Print Results
#                stats = trainlog(traj, agent=agent, diff_clf=agent.diff_clf, loss=0)
#                stats += f' traj: {traj}\n'
#                stats += f' num_steps: {num_steps}\n'
#                stats += f' ****************\n'
#                stats += f' Difference Classifier:\n'
#                stats += f'     Acuracy  : {diff_clf_stats["acur"]}\n'
#                stats += f'     F1 macro : {diff_clf_stats["f1_macro"]}\n'
#                stats += f'     Miss rate: {diff_clf_stats["miss_rate"]}\n'
#                stats += f'     (tn, fp, fn, tp): {tuple(diff_clf_stats["con_mat"])}\n'
#                stats += f' *****************\n'
#                stats += f' Model Classifier:\n'
#                stats += f'     Acuracy  : {model_stats["acur"]}\n'
#                stats += f'     F1       : {model_stats["f1"]}\n'
#                stats += f'     sklearnf1: {model_stats["sklearn_f1_macro"]}\n'
#                stats += f'     Train Reward: {ep_reward}\n'
#                stats += f'     Test Reward:  {model_stats["avg_rtn"]}\n'
#                stats += f' Train/Update: {timedelta(seconds=trainend-trainstart)}\n'
#                stats += f' Train/Update/Evaluate: {timedelta(seconds=allend-allstart)}\n'
#                stats += f' Between Update: {timedelta(seconds=nextupdateend-nextupdatestart)}\n'
#                stats += f'-----------------------------------'
#                print(stats)
#
#                # Log Results
#                results ={'x': num_steps, 'train_accuracy':ep_reward,\
#                     'test_accuracy':reward,\
#                     'expert_queries':trainlog.strongoracle_counts,\
#                     'ref_queries': trainlog.weakoracle_counts,\
#                     'no_queries': trainlog.nocall_counts,\
#                     'agent_random_queries': agent.random_queries,\
#                     'agent_seen_mistakes': agent.seen_mistakes,\
#                     'agent_total_mistakes': agent.total_mistakes,\
#                     'agent_unseen_mistakes': agent.unseen_mistakes,\
#                     'agent_other_mistakes': agent.other_mistakes,\
#                     'diff_clf_acur': diff_clf_stats["acur"],\
#                     'diff_clf_f1': diff_clf_stats['f1_macro'],\
#                     'tn': diff_clf_stats["con_mat"][0], 'fp': diff_clf_stats["con_mat"][1],\
#                     'fn': diff_clf_stats["con_mat"][2], 'tp': diff_clf_stats["con_mat"][3]}
#                for key in model_stats.keys():
#                    results[f'model_{key}'] = model_stats[key]
#
#                if args.alg == 'leaqi':
#                    for key in diff_clf_stats.keys():
#                        results[f'diff_{key}'] = diff_clf_stats[key]
#
#                data.append(results)
#                filename = args.filename if args.filename != 'tmp' else filename
#                pd.DataFrame(data).to_csv(f'{filename}_results.csv')
#                nextupdatestart = timer()
#            else:
#                allend = timer()
#                stats =  f' traj: {traj}\n'
#                stats += f' num_steps: {num_steps}\n'
#                stats += f' unseen_mistakes : {agent.unseen_mistakes}\n'
#                stats += f' model_mistakes: {agent.model_mistakes}\n'
#                stats += f' seen mistakes: {agent.seen_mistakes}\n'
#                stats += f' mistakes: {agent.mistakes}\n'
#                stats += f' total: {agent.total_mistakes}\n'
#                stats += f' Train/Update: {timedelta(seconds=trainend-trainstart)}\n'
#                stats += f' Train/Update/Evaluate: {timedelta(seconds=allend-allstart)}\n'
#                stats += f'-----------------------------------'
#                print(stats)
#
#
#            if 'stopiteration' in info:
#                if info['stopiteration']:
#                    break
#    except KeyboardInterrupt:
#        import pdb; pdb.set_trace()
#
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    wso_args(parser)
#    args = parser.parse_args()
#    print(args.filename)
#    main(args)
#
