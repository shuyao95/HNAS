import pickle
import argparse
import numpy as np
import os

import torch
from bayes_opt import BayesianOptimization
from nas_201_api import NASBench201API as API
from nats_bench import create

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='/Volumes/ZSSD/Research/NATS/tss/NATS-tss-v1_0-3ffb9-simple', type=str, help='path to API')
    parser.add_argument('--outdir', default='./', type=str, help='output directory')
    parser.add_argument('--dataset', type=str, default=None, help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--metric', type=str, default='ntk_trace', help="metric to use ['grad_norm', 'snip', 'grasp', 'ntk_trace']")
    
    parser.add_argument('--scale', type=float, default=5e3, help="")
    parser.add_argument('--n_sample', type=int, default=2000, help='pytorch manual seed')
    
    parser.add_argument('--total_iters', type=int, default=20, help='pytorch manual seed')
    parser.add_argument('--init_portion', type=float, default=0.25, help='pytorch manual seed')
    parser.add_argument('--acq', type=str, default='ei',help='choice of bo acquisition function, [ucb, ei, poi]')
    
    parser.add_argument('--hp', type=int, default=12, help='pytorch manual seed')
    
    parser.add_argument('--seed', type=int, default=0, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    args = parser.parse_args()
    return args

def get_infos(path, args):
    if os.path.exists(path):
        print('start loading infos....')
        with open(path, 'rb') as f:
            infos = pickle.load(f)
    else:
        api = create(args.api_loc, 'tss', fast_mode=True,verbose=False)
        infos = []
        for i, arch in enumerate(api):
            data = api.get_more_info(i, 'cifar10-valid', iepoch=None, hp=str(args.hp), is_random=False)
            infos += [{
                f'{i}'      : i,
                'arch'      : arch,
                'valacc'    : data['valid-accuracy'],
                'cost'      : data['train-all-time'],
            }]
        print('start saving infos....')
        with open(path, 'wb') as f:
            pickle.dump(infos, f)
    return infos

def score_arch(metric, target_metric, condnum, mu=1): # a good suggestion is that mu = 1 / max_metric ** 2
    # score = mu * abs(metric**2 - target_metric**2) + condnum / (metric + 1e-6)
    score = 1 / (metric + 1e-6)
    return score

def search(target_metric, target_coeff, datas, args):
    target_metric = target_metric * args.scale
    
    data_metrics, data_condnums, test_accs = datas
    global sampled_archs, opt_archs
    sampled_archs = np.random.choice(len(data_metrics), args.n_sample)
    sampled_archs = list(dict.fromkeys(sampled_archs)) # remove redundent
    
    stats, condnums = [], []
    for arch in sampled_archs:
        stats += [data_metrics[arch]]
        condnums += [data_condnums[arch]]
        
    scores = [score_arch(m, target_metric, c, 0) for m, c in zip(stats, condnums)]
    opt_archs = [sampled_archs[idx.item()] for idx in np.argsort(scores)]
    
    
if __name__ == '__main__':
    args = parse_arguments()
    np.random.seed(args.seed)
    
    # load the statistics
    file = "visualize/stats/nb2_cf10_seed42_initwnone.p"
    with open(file, 'rb') as f:
        data_stats = pickle.load(f)
    data_metrics, data_condnums, data_inds = [], [], []
    for data in data_stats:
        keys = data['logmeasures'].keys()
        if args.metric in keys and 'condnum' in keys:
            data_inds += [data['i']]
            data_metrics += [abs(data['logmeasures'][args.metric])]
            data_condnums += [data['logmeasures']['condnum']]
    
    # load accuracy
    file = "data/nasbench2_accuracy.p"
    with open(file, 'rb') as f:
        data_accs = pickle.load(f)
    
    if args.dataset:
        test_accs = [data_accs[ind][args.dataset]['testacc'] for ind in data_inds]    
    else:
        test_accs = [
            (
                data_accs[ind]['cifar10']['testacc'].item(), 
                data_accs[ind]['cifar100']['testacc'].item(), 
                data_accs[ind]['ImageNet16-120']['testacc'].item()
            )
            for ind in data_inds
        ]    
    # load info
    infos = get_infos('data/nasbench2_info_hp%d.p' % (args.hp), args)
    val_accs = [infos[ind]['valacc'] for ind in data_inds]
    
    datas = [data_metrics, data_condnums, val_accs]
    
    # the domain for search
    pbounds = {
        'target_metric': (0, 1),
        'target_coeff': (-4, 0),
    }
    
    sampled_archs = [] 
    opt_archs = []

    search(0, 0, datas, args)
    budgets = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
    budget_ids = [b // 120 for b in budgets]
    
    # to obtain the search cost
    cost = 0
    for i, arch in enumerate(opt_archs[:400]):
        cost += infos[data_inds[arch]]['cost']
        final_ind = opt_archs[np.argmax([val_accs[arch] for arch in opt_archs[:i+1]]).item()]
        true_opt_archs = [data_inds[arch] for arch in opt_archs[:i+1]]
        
        # print(true_opt_archs)
        if i+1 in budget_ids:
            print(','.join(map(lambda x: "%.2f" % x, test_accs[final_ind])))
    