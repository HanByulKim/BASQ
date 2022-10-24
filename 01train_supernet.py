
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
from mbv2_supernet import generate_efficient_mbv2, QtActivation, InvertedResidual, lastconv1x1, ActLsqQuan

parser = argparse.ArgumentParser("mbv2")
parser.add_argument('--batch_size', type=int, default=768, help='batch size')
parser.add_argument('--abin_total_iters', type=int, default=466000, help='num of training epochs')
parser.add_argument('--awbin_total_iters', type=int, default=466000, help='num of training epochs')
parser.add_argument('--save_interval', type=int, default=1000, help='num of training epochs')
parser.add_argument('--bits', type=int, default=2, help='num bits')
parser.add_argument('--learning_rate_abin', type=float, default=3.75e-4, help='init learning rate')
parser.add_argument('--learning_rate_wbin', type=float, default=3.75e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay_abin', type=float, default=5e-6, help='weight decay from AdamBNN paper')
parser.add_argument('--weight_decay_wbin', type=float, default=5e-6, help='weight decay')
parser.add_argument('--l2_bounds', type=list, default=[0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001], help='weight decay')
parser.add_argument('--depth_multiplier', type=float, default=1)
parser.add_argument('--width_multiplier', type=float, default=1)
parser.add_argument('--width_divisor', type=int, default=1)
parser.add_argument('--resolution_multiplier', type=float, default=1.0)#1.15
parser.add_argument('--base_resolution', type=int, default=224)
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', type=str, default='/SSD/ILSVRC2012')
parser.add_argument('--teacher', type=str, default='resnet50', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')                    
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
args = parser.parse_args()

CLASSES = 1000
all_iters = 0

args.save = 'train-supernet'
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '{%(asctime)s}-(%(process)d)-%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
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


def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    # load model
    model_teacher = None
    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    model_student, resolution = generate_efficient_mbv2(CLASSES, args.width_multiplier, args.width_divisor, args.depth_multiplier, args.resolution_multiplier, args.base_resolution, n_lv=(2**args.bits), l2_vals=args.l2_bounds, bit=args.bits)
    logging.info('student:')
    logging.info(model_student)
    chklth(model_student)

    model_path = None
    if model_path:
        model_student.load_model(model_path)
    else:
        model_student = nn.DataParallel(model_student)
    model_student = model_student.cuda()

    criterion_kd = KD_loss.DistributionLoss()
    criterion_kd = criterion_kd.cuda()

    start_iter = 0
    best_top1_iter = 0
    best_top1_acc= 0

    # load training data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # data augmentation
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

    ## train from scratch
    ## abin train
    weight_parameters = []
    bin_parameters = []
    bound_parameters = [[] for i in range(len(args.l2_bounds))]
    other_parameters = []
    for pname, p in model_student.named_parameters():
        print(pname)
        if 'upper_bound' in pname:
            for i in range(19):
                if f'feature.{i}.' in pname:
                    for j in range(len(args.l2_bounds)):
                        if f'branch_q_0.{j}' in pname or f'branch_bn_q_1.{j}' in pname or f'branch_q_2.{j}' in pname:
                            print(f"pname1 {pname}-{i}-{j}")
                            bound_parameters[j].append(p)
                            # print(f"upper_bound {pname}")
        elif 'bin' in pname:
            bin_parameters.append(p)
            # print(f"bin_parameters {pname}")
        elif p.ndimension() == 4 or 'weights' in pname:
            weight_parameters.append(p)
            # print(f"weight_parameters {pname}")
        else:
            other_parameters.append(p)
            # print(f"other_parameters {pname}")

    print([{'params' : bound_parameters[i], 'weight_decay' : args.l2_bounds[i]} for i in range(len(args.l2_bounds))])
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay_abin}]
            + [{'params' : bound_parameters[i], 'weight_decay' : args.l2_bounds[i]} for i in range(len(args.l2_bounds))],
            lr=args.learning_rate_abin,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.abin_total_iters)
    best_top1_iter = 0
    best_top1_acc= 0
    start_iter = 0
    
    if model_path:
        chkpt = torch.load(model_path)
        start_iter = chkpt['iter']+1
        for i in range(start_iter):
            scheduler.step()

    # train the model
    model_student.module.change_precision(a_bin=True, w_bin=False)
    if model_path is None:
        initialize(model_student, train_loader)
    chklth(model_student)

    global all_iters
    all_iters = start_iter
    logging.info(f'==actbin-({all_iters}/{args.abin_total_iters}) lr-({scheduler.get_lr()})==')
    while all_iters < args.abin_total_iters:
        tic = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(args.abin_total_iters, train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler, w_bin=False)

        logging.info(f"Epoch time: {time.time()-tic}")

    training_time = (time.time() - start_t) / 3600
    print('actbin total training time = {} hours'.format(training_time))

    ## aw_bin train
    print([{'params' : bound_parameters[i], 'weight_decay' : args.l2_bounds[i]} for i in range(len(args.l2_bounds))])        
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay_wbin}]
            + [{'params' : bound_parameters[i], 'weight_decay' : args.l2_bounds[i]} for i in range(len(args.l2_bounds))],
            lr=args.learning_rate_wbin,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.awbin_total_iters)
    best_top1_iter = 0
    best_top1_acc= 0

    # train the model
    model_student.module.change_precision(a_bin=True, w_bin=True)
    model_student.module.reinit_conv()
    chklth(model_student)
    all_iters = 0
            
    logging.info(f'==actbinwbin_iter-({all_iters}/{args.awbin_total_iters}) lr-({scheduler.get_lr()})==')
    while all_iters < args.awbin_total_iters:
        tic = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(args.awbin_total_iters, train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler, w_bin=True)

        logging.info(f"Iter time: {time.time()-tic}")

    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours'.format(training_time))

def train(total_iters, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, w_bin=None):
    global all_iters
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model_student.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        all_iters += 1
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        get_random_selection = lambda:tuple(np.random.randint(len(args.l2_bounds)) for i in range(18))

        # compute outputy
        random_selection = get_random_selection()
        logits_student = model_student(images, random_selection)
        logits_teacher = model_teacher(images)

        n = images.size(0)
        loss = criterion(logits_student, logits_teacher)
        losses.update(loss.item(), n)   #accumulated loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_student.parameters(), 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if all_iters % args.report_freq == 0:
            logging.info(f'train: ({all_iters:03d}/{total_iters:03d})-({i:03d}/{len(train_loader):03d})-(objs:{losses.avg:0.3f})-(acc:{top1.avg:0.3f}))')

        if all_iters % 1000 ==0:
            logging.info(f'==iter-({all_iters}/{total_iters}) lr-({scheduler.get_lr()})==')
            logging.info(f'selection={random_selection}')
            chklth(model_student)

        if all_iters % args.save_interval == 0 or all_iters==1:
            torch.save({
                'iter': all_iters,
                'model_state_dict': model_student.module.state_dict(),
                'best_top1_acc': top1.avg,
            }, os.path.join(args.save, f'model_actwbin_best_{all_iters:05d}it.pth' if w_bin else f'model_actbin_best_{all_iters:05d}it.pth'))

        scheduler.step()
    return losses.avg, top1.avg, top5.avg

def initialize(model, loader):
    def initialize_hook(module, input, output):
        if isinstance(module, ActLsqQuan):
            if not isinstance(input, torch.Tensor):
                input = input[0]
            module.init_from(input)

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)
    
    model.train()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            get_random_selection = lambda:tuple(np.random.randint(len(args.l2_bounds)) for i in range(18))
            output = model(input, get_random_selection())
        break
    
    for hook in hooks:
        hook.remove()


if __name__ == '__main__':
    main()
