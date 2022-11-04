import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import pickle

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

from scipy.special import softmax

import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from bayes_opt import BayesianOptimization
import darts_space.utils as utils
import tools.autograd_hacks as autograd_hacks
from darts_space.genotypes import *



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--arch_path', type=str, default='data/sampled_archs.p', help='location of the data corpus')
parser.add_argument('--no_search', action='store_true',default=False, help='only apply sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for search')
parser.add_argument('--batch_size', type=int, default=576, help='batch size')
parser.add_argument('--metric_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

parser.add_argument('--scale', type=float, default=1e2, help="")
parser.add_argument('--n_sample', type=int, default=60000, help='pytorch manual seed')

parser.add_argument('--total_iters', type=int, default=25, help='pytorch manual seed')
parser.add_argument('--init_portion', type=float, default=0.25, help='pytorch manual seed')
parser.add_argument('--acq', type=str, default='ucb',help='choice of bo acquisition function, [ucb, ei, poi]')
args = parser.parse_args()
args.cutout = False
args.auxiliary = False

args.save = 'darts/search-{}-{}'.format(args.save, args.dataset)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, f'S{args.seed}-log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
    NUM_CLASSES = 10
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'cifar100':
    NUM_CLASSES = 100
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'imagenet':
    NUM_CLASSES = 1000
    from darts_space.model import NetworkImageNet as Network
else:
    raise ValueError('Donot support dataset %s' % args.dataset)
    
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # load the dataset
    if 'cifar' in args.dataset:
        train_transform, valid_transform = eval("utils._data_transforms_%s" % args.dataset)(args)
        train_data = eval("dset.%s" % args.dataset.upper())(
            root=args.data, train=True, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=4)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]),
            pin_memory=True, num_workers=4)
    elif 'imagenet' in args.dataset:
        train_queue, valid_queue = utils._get_imagenet(args)
    else:
        raise ValueError("Donot support dataset %s" % args.dataset)

    data_queues = [train_queue, valid_queue]

    # the domain for search
    pbounds = {
        'target_metric': (0, 1),
        'target_coeff': (-4, 1),
    }
    
    
    global_vars = get_pool(data_queues, args)
    
    if not args.no_search:
        
        optimizer = BayesianOptimization(
            f               = lambda target_metric, target_coeff: search(global_vars, [target_metric, target_coeff], data_queues, args), 
            pbounds         = pbounds,
            random_state    = args.seed
        )
        
        start = time.time()
        init_points = min(int(args.total_iters * args.init_portion), 10)
        if init_points > 0:
            probe_points = np.linspace(0, 1, init_points)
            for i in range(init_points):
                optimizer.probe(
                    params={'target_metric': probe_points[i].item(), 'target_coeff': np.mean(pbounds['target_coeff']).item()},
                    lazy=True,
                )

        optimizer.maximize(
            init_points     = 0,
            n_iter          = args.total_iters - init_points,
            acq             = args.acq,
        )
        logging.info('Search cost = %.2f(h)' % ((time.time() - start) / 3600, ))
        
        sorted_orders = sorted(global_vars[1].items(), key=lambda x: x[1][-1], reverse=True)
        logging.info('Genotype = %s' % (sorted_orders[0][0], ))


def get_pool(data_queues, args):
    size=[14 * 2, 7]
    train_queue, _ = data_queues
    
    if not os.path.exists(args.arch_path):
        start = time.time()
        logging.info('Start sampling architectures...')
        
        sampled_genos, opt_genos, sampled_metrics = {}, {}, {}
        
        new_weights = [np.random.random_sample(size) for _ in range(args.n_sample)]
        new_genos = [genotype(w.reshape(2, -1, size[-1])) for w in new_weights]
        new_keys = list(map(str, new_genos))
        
        sampled_genos = dict(zip(new_keys, new_genos))
        
        inputs, targets = next(iter(train_queue))
        inputs, targets = inputs[:args.metric_batch_size].cuda(), targets[:args.metric_batch_size].cuda()
        
        # compute the training-free metrics
        for i, (k, geno) in enumerate(sampled_genos.items()):
            if i % 1000 == 0:
                logging.info('Start computing the metrics for arch %06d' % (i, ))
            
            model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, geno).cuda()
            model.drop_path_prob = 0
            m, c = compute_metrics(model, inputs, targets)
            sampled_metrics.update({k: (m, c)})
            
        with open(args.arch_path, 'wb') as f:
            pickle.dump([sampled_genos, opt_genos, sampled_metrics], f)
            
        logging.info('Sampling cost=%.2f(h)' % ((time.time()- start) / 3600, ))
    else:
        with open(args.arch_path, 'rb') as f:
            sampled_genos, opt_genos, sampled_metrics = pickle.load(f)
    
    return sampled_genos, opt_genos, sampled_metrics


def search(global_vars, bo_hyper_params, data_queues, args, size=[14 * 2, 7]):
    sampled_genos, opt_genos, sampled_metrics = global_vars
    train_queue, _ = data_queues
    
    # remove evaluated archs
    sampled_genos = {k:g for k, g in sampled_genos.items() if k not in opt_genos.keys()}
    
    # training-free search
    target_metric, target_coeff = bo_hyper_params
    mu = 10 ** target_coeff
    target_metric = target_metric * args.scale
    
    sampled_keys = sampled_genos.keys()
    scores = [score_arch(sampled_metrics[k][0], target_metric, sampled_metrics[k][1], mu) for k in sampled_keys]
    opt_key = list(sampled_keys)[np.argsort(scores)[0].item()]
    opt_geno = sampled_genos[opt_key]
    
    # training-based search
    opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
    val_acc = train(data_queues, opt_model)
    opt_genos.update({opt_key: (opt_geno, val_acc)})
    
    logging.info('Genotype = %s' % (opt_geno, ))
    return val_acc


def score_arch(metric, target_metric, condnum, mu=1): # a good suggestion is that mu = 1 / max_metric ** 2
    score = mu * abs(metric**2 - target_metric**2) + condnum / (metric + 1e-6)
    return score

def compute_metrics(net, inputs, targets):
    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    grads = []
    
    autograd_hacks.add_hooks(net)
    outputs, _ = net.forward(inputs)
    sum(outputs[torch.arange(N), targets]).backward()
    autograd_hacks.compute_grad1(net, loss_type='sum')
    
    grads = [param.grad1.flatten(start_dim=1) for param in net.parameters() if hasattr(param, 'grad1')]
    grads = torch.cat(grads, axis=1)
    
    ntk = torch.matmul(grads, grads.t())
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    return np.sqrt(eigenvalues.cpu().numpy().sum() / N), np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)


def train(data_queues, model):
    train_queue, valid_queue = data_queues
    
    if 'imagenet' in args.dataset:
        criterion = utils.CrossEntropyLabelSmooth(NUM_CLASSES, args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    model.train()
    
    for epoch in range(args.epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        logging.info('epoch %d lr %e drop_prob %e', epoch, scheduler.get_last_lr()[0], model.drop_path_prob)

        for step, (input, target) in enumerate(train_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        
        scheduler.step()

    # validation
    valid_acc = infer(valid_queue, model, criterion)

    return valid_acc


def infer(valid_queue, model, criterion):
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %f', step, top1.avg)

    return top1.avg

def genotype(weights, steps=4, multiplier=4):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(
                W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene
        
    gene_normal = _parse(softmax(weights[0], axis=-1))
    gene_reduce = _parse(softmax(weights[1], axis=-1))

    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

if __name__ == '__main__':
    main()