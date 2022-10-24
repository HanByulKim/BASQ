import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random
import logging

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

from mbv2_supernet import generate_efficient_mbv2, InvertedResidual, lastconv1x1

from torch.autograd import Variable
from torchvision import datasets, transforms
import collections
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

from utils.utils import *

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

CLASSES = 1000

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--data', type=str, default='/SSD/ILSVRC2012')

parser.add_argument('--k', type=int, default=10, help='k')
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--select-num', type=int, default=10)
parser.add_argument('--population-num', type=int, default=45)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=15)
parser.add_argument('--mutation-num', type=int, default=15)
parser.add_argument('--max_val_iters', type=int, default=25)
parser.add_argument('--val_batch_size', type=int, default=200)

parser.add_argument('--bits', type=int, default=2, help='num bits')
parser.add_argument('--l2_bounds', type=list, default=[0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001], help='weight decay')
parser.add_argument('--depth_multiplier', type=float, default=1)
parser.add_argument('--width_multiplier', type=float, default=1)
parser.add_argument('--width_divisor', type=int, default=1)
parser.add_argument('--resolution_multiplier', type=float, default=1.0)#1.15
parser.add_argument('--base_resolution', type=int, default=224)
args = parser.parse_args()

args.save = 'eval-kfold-evolution'
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '{%(asctime)s}-(%(process)d)-%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class EvolutionSearcher(object):

    def __init__(self, k, args, val_loader, model_path=None):
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num

        self.val_loader = val_loader
        self.model, _ = generate_efficient_mbv2(CLASSES, args.width_multiplier, args.width_divisor, args.depth_multiplier, args.resolution_multiplier, args.base_resolution, n_lv=(2**args.bits), l2_vals=args.l2_bounds, bit=args.bits)
        self.model.load_model(model_path)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.module.change_precision(a_bin=True, w_bin=True)

        self.checkpoint_name = os.path.join(args.save, f'checkpoint_{k}.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 17
        self.nr_state = len(args.l2_bounds)

    def delete_elems(self):
        del self.model
        del self.memory
        del self.vis_dict
        del self.keep_top_k
        del self.candidates

    def chklth(self):
        for m in model.modules():
            if isinstance(m, InvertedResidual):
                if m.expand_ratio != 1:
                    print(f"[{m.index}]-1x1: LSQ")

                ubs = [branch[1].upper_bound.item() for branch in m.branch_bn_q_1]
                print(f"[{m.index}]-3x3: n_lv-{m.n_lv}, clip-[", end="")
                for ub in ubs: print(f"{ub:.3f},", end="")
                print(f"]")

                print(f"[{m.index}]-1x1: LSQ")

            if isinstance(m, lastconv1x1):
                print(f"[{m.index}]-1x1: LSQ")

    def save_checkpoint(self):
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        logging.info(f'save checkpoint to {self.checkpoint_name}')

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        logging.info(f'load checkpoint from {self.checkpoint_name}')
        return True

    def is_legal(self, cand, kfold):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        info['err'] = validate_selections(kfold, list(cand), self.val_loader, self.model, self.args.max_val_iters, full=False)
        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num, kfold):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand, kfold):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, kfold):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand, kfold):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num, kfold):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand, kfold):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self, kfold):
        print('fold = {}, max_iter = {}, population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format( kfold, self.args.max_val_iters,
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.get_random(self.population_num, kfold)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                if i < 1:
                    self.vis_dict[cand]['full_acc'] = 100 - validate_selections(kfold, list(cand), self.val_loader, self.model, self.args.max_val_iters, full=True)
                else:
                    self.vis_dict[cand]['full_acc'] = 0
                print('No.{} {} Top-1 err = {}, Full top-1 = {:.2f}'.format(i + 1, cand, self.vis_dict[cand]['err'], self.vis_dict[cand]['full_acc']))
                ops = [i for i in cand]
                print(ops)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob, kfold)
            crossover = self.get_crossover(self.select_num, self.crossover_num, kfold)

            self.candidates = mutation + crossover

            self.get_random(self.population_num, kfold)

            self.epoch += 1

        self.save_checkpoint()

        self.memory.append([])
        for cand in self.candidates:
            self.memory[-1].append(cand)

        self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
        self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

        fold_acc = 0
        full_acc = 0
        arch = []
        print('**Final top {} result'.format(len(self.keep_top_k[50])))
        for i, cand in enumerate(self.keep_top_k[50]):
            if i < 1:
                self.vis_dict[cand]['full_acc'] = 100 - validate_selections(kfold, list(cand), self.val_loader, self.model, self.args.max_val_iters, full=True)
                fold_acc = 100 - self.vis_dict[cand]['err']
                full_acc = self.vis_dict[cand]['full_acc']
                arch = cand
            else:
                self.vis_dict[cand]['full_acc'] = 0
            print('No.{} {} Top-1 err = {}, Full top-1 = {:.2f}'.format(i + 1, cand, self.vis_dict[cand]['err'], self.vis_dict[cand]['full_acc']))
            ops = [i for i in cand]
            print(ops)

        return fold_acc, full_acc, arch        

def validate_selections(k, selections, val_loader, model, max_val_iters, full=False):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    runs = []

    # switch to evaluation mode
    tic = time.time()
    model.eval()
    with torch.no_grad():
        end = time.time()
        if full:
            for i, (images, target) in enumerate(val_loader):
                if (i >= k*max_val_iters) and (i < (k+1)*max_val_iters): continue
                images = images.cuda()
                target = target.cuda()

                logits = model(images, selections)

                n = images.size(0)
                pred1, _ = accuracy(logits, target, topk=(1, 5))
                top1.update(pred1.item(), n)
                batch_time.update(time.time() - end)
                end = time.time()
                runs.append(i)
        else:
            for i, (images, target) in enumerate(val_loader):
                if (i >= (k+1)*max_val_iters): break
                if (i >= k*max_val_iters) and (i < (k+1)*max_val_iters):
                    images = images.cuda()
                    target = target.cuda()

                    logits = model(images, selections)

                    n = images.size(0)
                    pred1, _ = accuracy(logits, target, topk=(1, 5))
                    top1.update(pred1.item(), n)
                    batch_time.update(time.time() - end)
                    end = time.time()
                    runs.append(i)

        logging.info(f'{selections}-[{runs[0]},{runs[-1]}]-({max_val_iters}/{len(val_loader)})-(time:{time.time()-tic:.3f})-(acc:{top1.avg:.3f}))')
        
    return 100 - top1.avg

def main():

    t = time.time()

    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    args.max_val_iters = len(val_loader) // args.k

    model_path = os.path.join('./train-supernet', 'model_actwbin_best_466000it.pth')

    fold_accs = []
    full_accs = []
    archs = []
    for i in range(args.k):
        logging.info(f"===fold-({i})===")
        searcher = EvolutionSearcher(i, args, val_loader, model_path=model_path)
        fold_acc, full_acc, arch = searcher.search(i)
        fold_accs.append(fold_acc)
        full_accs.append(full_acc)
        archs.append(arch)
        logging.info(f"===fold-({i})-result===")
        logging.info([k for k in range(i+1)])
        logging.info(f"fold_accs: {fold_accs}")
        logging.info(f"full_accs: {full_accs}")
        logging.info(f"archs: {archs}")

    logging.info(f"===final-result===")
    logging.info([k for k in range(args.k)])
    logging.info(f"fold_accs: {fold_accs}")
    logging.info(f"full_accs: {full_accs}")
    logging.info(f"archs: {archs}")
    print('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
