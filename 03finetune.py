
import os
import sys
import shutil
import numpy as np
import time, datetime
import glob
import torch
import random
import logging
import argparse
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

from utils.utils import *
from utils import utils
from utils import KD_loss
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
from mbv2_supernet import generate_efficient_mbv2, InvertedResidual, lastconv1x1

parser = argparse.ArgumentParser("mbv2")
parser.add_argument('--batch_size', type=int, default=768, help='batch size')
parser.add_argument('--bn_epochs', type=int, default=3, help='num of training epochs')
parser.add_argument('--bits', type=int, default=2, help='num bits')
parser.add_argument('--learning_rate_bnstabilize', type=float, default=7.5e-6, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--l2_bounds', type=list, default=[0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001], help='weight decay')
parser.add_argument('--depth_multiplier', type=float, default=1)
parser.add_argument('--width_multiplier', type=float, default=1)
parser.add_argument('--width_divisor', type=int, default=1)
parser.add_argument('--resolution_multiplier', type=float, default=1.0)#1.15
parser.add_argument('--base_resolution', type=int, default=224)
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', type=str, default='/SSD/ILSVRC2012')
parser.add_argument('--teacher', type=str, default='resnet50', help='path of ImageNet') #resnet101 also
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--train_clip', type=bool, default=True, help='train_clip')

parser.add_argument('--k', type=int, default=10)
parser.add_argument('--max_val_iters', type=int, default=25)
parser.add_argument('--val_batch_size', type=int, default=200)
args = parser.parse_args()

CLASSES = 1000
all_iters = 0

## Results from 02kfold_evolution.py ##
# Please paste final results printed in 02kfold_evolution.py
# For example, our results are same as below.
# {02/22 06:44:32 AM}-(2484025)-===total-result===
# {02/22 06:44:32 AM}-(2484025)-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# {02/22 06:44:32 AM}-(2484025)-fold_accs: [67.62, 66.2, 64.86, 68.94, 55.18, 58.96, 54.82, 56.46, 53.54, 63.48]
# {02/22 06:44:32 AM}-(2484025)-full_accs: [59.30222222222222, 59.29333333333334, 59.26222222222222, 58.67777777777778, 59.733333333333334, 59.15777777777778, 60.01111111111111, 60.45333333333333, 60.10888888888889, 59.53333333333333]
# {02/22 06:44:32 AM}-(2484025)-archs: [(5, 3, 2, 2, 1, 1, 6, 6, 1, 2, 4, 6, 0, 3, 3, 5, 5), (1, 4, 5, 0, 5, 2, 3, 6, 4, 5, 6, 6, 6, 6, 3, 2, 5), (1, 4, 5, 0, 2, 5, 5, 2, 5, 2, 1, 1, 2, 2, 6, 4, 5), (1, 4, 5, 1, 1, 2, 4, 6, 2, 1, 2, 6, 4, 2, 6, 4, 5), (1, 4, 5, 5, 1, 5, 1, 5, 6, 5, 4, 1, 6, 4, 3, 2, 3), (5, 3, 2, 2, 6, 6, 5, 0, 5, 5, 6, 6, 4, 6, 5, 0, 2), (1, 4, 5, 1, 6, 2, 6, 4, 2, 1, 1, 6, 4, 0, 6, 4, 5), (5, 3, 2, 2, 1, 1, 4, 2, 1, 2, 4, 6, 0, 0, 3, 6, 5), (1, 4, 5, 5, 6, 5, 2, 3, 5, 6, 3, 6, 5, 2, 6, 0, 2), (5, 3, 2, 2, 4, 1, 4, 2, 1, 4, 6, 6, 0, 4, 3, 2, 5)]


FOLD_ACCS = [67.62, 66.2, 64.86, 68.94, 55.18, 58.96, 54.82, 56.46, 53.54, 63.48]
FULL_ACCS = [59.30222222222222, 59.29333333333334, 59.26222222222222, 58.67777777777778, 59.733333333333334, 59.15777777777778, 60.01111111111111, 60.45333333333333, 60.10888888888889, 59.53333333333333]
BEST_ARCHS = [(5, 3, 2, 2, 1, 1, 6, 6, 1, 2, 4, 6, 0, 3, 3, 5, 5), (1, 4, 5, 0, 5, 2, 3, 6, 4, 5, 6, 6, 6, 6, 3, 2, 5), (1, 4, 5, 0, 2, 5, 5, 2, 5, 2, 1, 1, 2, 2, 6, 4, 5), (1, 4, 5, 1, 1, 2, 4, 6, 2, 1, 2, 6, 4, 2, 6, 4, 5), (1, 4, 5, 5, 1, 5, 1, 5, 6, 5, 4, 1, 6, 4, 3, 2, 3), (5, 3, 2, 2, 6, 6, 5, 0, 5, 5, 6, 6, 4, 6, 5, 0, 2), (1, 4, 5, 1, 6, 2, 6, 4, 2, 1, 1, 6, 4, 0, 6, 4, 5), (5, 3, 2, 2, 1, 1, 4, 2, 1, 2, 4, 6, 0, 0, 3, 6, 5), (1, 4, 5, 5, 6, 5, 2, 3, 5, 6, 3, 6, 5, 2, 6, 0, 2), (5, 3, 2, 2, 4, 1, 4, 2, 1, 4, 6, 6, 0, 4, 3, 2, 5)]

args.save = 'eval-blast'
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '{%(asctime)s}-(%(process)d)-%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def chklth(model):
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

def categorize_param(model, skip_list=()):
    skip_pararmeters = []
    bound_parameters = [[] for i in range(len(args.l2_bounds))]
    bnbias_pararmeters = []
    bin_parameters = []
    weight_parameters = []
    other_parameters = []
    for pname, p in model.named_parameters():
        skip_found = False
        for s in skip_list:
            if pname.find(s) != -1:
                skip_found = True

        if skip_found:
            skip_pararmeters.append(p)
            print(f"skip - {pname}")
        elif 'upper_bound' in pname:
            for i in range(len(args.l2_bounds)):
                if f'branch_q_0.{i}' in pname or f'branch_bn_q_1.{i}' in pname or f'branch_q_2.{i}' in pname:
                    print(f"bound {pname}-{i}")
                    bound_parameters[i].append(p)
                    # print(f"upper_bound {pname}")
        elif 'bias' in pname:
            bnbias_pararmeters.append(p)
            print(f"bias - {pname}")
        elif 'bin' in pname:
            bin_parameters.append(p)
            # print(f"bin_parameters {pname}")
        elif p.ndimension() == 4 or 'weights' in pname:
            weight_parameters.append(p)
            # print(f"weight_parameters {pname}")
        else:
            other_parameters.append(p)
            # print(f"other_parameters {pname}")

    return (skip_pararmeters, bound_parameters, bnbias_pararmeters, bin_parameters, weight_parameters, other_parameters)

def get_optimizer(params, train_weight, train_bnbias, train_clip, lr=args.learning_rate_bnstabilize):
    (skip_pararmeters, bound_parameters, bnbias_pararmeters, bin_parameters, weight_parameters, other_parameters) = params
    optimizer = torch.optim.SGD([
        {'params': skip_pararmeters, 'weight_decay': 0, 'lr': 0},
        {'params': bnbias_pararmeters, 'weight_decay': 0., 'lr': lr if train_bnbias else 0},
        {'params': bin_parameters, 'weight_decay': 0., 'lr': 0},
        {'params': weight_parameters, 'weight_decay': 0., 'lr': lr if train_weight else 0},
        {'params': other_parameters, 'weight_decay': 0., 'lr': lr if train_weight else 0},
    ] + [{'params' : bound_parameters[i], 'weight_decay' : args.l2_bounds[i], 'lr': lr if train_clip else 0} for i in range(len(args.l2_bounds))],
    momentum=0.9, nesterov=True)
    return optimizer

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    # load model
    model_teacher = None
    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_kd = KD_loss.DistributionLoss()
    criterion_kd = criterion_kd.cuda()

    # load training data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    args.max_val_iters = len(val_loader)//args.k
    print(f"max_val_iters={args.max_val_iters}")
    best_accs = []

    ## aw_bin train
    for i, s in enumerate(BEST_ARCHS):
        model_student, resolution = generate_efficient_mbv2(CLASSES, args.width_multiplier, args.width_divisor, args.depth_multiplier, args.resolution_multiplier, args.base_resolution, n_lv=(2**args.bits), l2_vals=args.l2_bounds, bit=args.bits)
        model_path = os.path.join('./train-supernet', 'model_actwbin_best_466000it.pth')
        if model_path:
            model_student.load_model(model_path)
        model_student = nn.DataParallel(model_student).cuda()

        logging.info(f"==> ARCH[{i}] AWbin-{args.bits}bit BN stabilize [{s}]")
        model_student.module.change_precision(a_bin=True, w_bin=True)
        logging.info(f"==> ARCH[{i}]: fold_acc={FOLD_ACCS[i]:.2f}, full_acc={FULL_ACCS[i]:.2f}")
        _ = validate_selections(i, s, val_loader, model_student, args.max_val_iters, full=True)
        optimizer = get_optimizer(categorize_param(model_student), train_weight=False, train_bnbias=True, train_clip=args.train_clip)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.bn_epochs)
        logging.info(f"==> ARCH[{i}] AWbin-{args.bits}bit BN stabilize [{s}]")
        logging.info(f"==> ARCH[{i}]: fold_acc={FOLD_ACCS[i]:.2f}, full_acc={FULL_ACCS[i]:.2f}")
        best_acc = train_epochs(args.bn_epochs, i, s, train_loader, val_loader, model_student, model_teacher, criterion, criterion_kd, optimizer, scheduler, args, w_bit=args.bits)
        best_accs.append(best_acc)
        logging.info(f"*bestaccs: {best_accs}")
    
    logging.info(f"===final-result===")
    logging.info(f"fold_accs: {FOLD_ACCS}")
    logging.info(f"full_accs: {FULL_ACCS}")
    logging.info(f"**bnfull_accs: {best_accs}")
    logging.info(f"archs: {BEST_ARCHS}")

def train_epochs(epochs, idx, selection, train_loader, val_loader, model_student, model_teacher, criterion, criterion_kd, optimizer, scheduler, args, w_bit=None):

    best_epoch = 0
    best_acc = FULL_ACCS[idx]
    for epoch in range(epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model_student.train()
        model_teacher.eval()
        end = time.time()
        
        logging.info(f'==ARCH[{idx}]: BN_stabilizer-({epoch}/{epochs}) lr-({scheduler.get_lr()}) [{selection}]==')
        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            logits_student = model_student(images, selection)

            n = images.size(0)
            loss = criterion(logits_student, target)
            losses.update(loss.item(), n)
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 10)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.report_freq == 0 or i == len(train_loader)-1:
                logging.info(f'train: ({epoch:03d}/{epochs:03d})-({i:03d}/{len(train_loader):03d})-(objs:{losses.avg:0.3f})-(acc:{top1.avg:0.3f}))')
        val_acc = validate_selections(idx, selection, val_loader, model_student, args.max_val_iters, full=True)
        logging.info(f"==> ARCH[{idx}]: fold_acc={FOLD_ACCS[idx]:.2f}, full_acc={FULL_ACCS[idx]:.2f}")

        scheduler.step()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_student.module.state_dict(),
            'best_top1_acc': val_acc,
        }, os.path.join(args.save, f'arch{idx}_model_actwbin_{args.bits}_{w_bit}_bn_{epoch}ep.pth' if w_bit else f'arch{idx}_model_actbin_{args.bits}_{w_bit}_bn_{epoch}ep.pth'))
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
    return best_acc

def validate_selections(k, selections, val_loader, model, max_val_iters, full=True):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    runs = []
    noruns = []

    # switch to evaluation mode
    tic = time.time()
    model.eval()
    with torch.no_grad():
        end = time.time()
        if full:
            for i, (images, target) in enumerate(val_loader):
                if (i >= k*max_val_iters) and (i < (k+1)*max_val_iters):
                    noruns.append(i)
                    continue
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
            noruns.append(-1)
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
        logging.info(f'{selections}-fold[{noruns[0]},{noruns[-1]}]-[{runs[0]},{runs[-1]}]-({max_val_iters}/{len(val_loader)})-(time:{time.time()-tic:.3f})-(acc:{top1.avg:.3f}))')
        
    return top1.avg


if __name__ == '__main__':
    main()
