#
import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from imbalance_fashion import IMBALANCEFASHION
from losses import LDAMLoss, FocalLoss, SBLoss,EQL,Cyclical_FocalLoss,LMFLoss
from opts import parser
from sam import SAM
from scipy.interpolate import interp1d
best_acc1 = 0
import warnings
warnings.filterwarnings("ignore")
import argparse
import models



def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    print(args.store_name)
    prepare_folders(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(base_optimizer=base_optimizer, rho=args.rho, params=model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)

    # minimizer = eval("ASAM")(optimizer, model)
    #
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:

                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True 

    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'fashion':
        train_dataset = IMBALANCEFASHION(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    if args.loss_type=="EQL":
        cls_per=np.array(cls_num_list)/sum(cls_num_list)
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        adjust_rho(optimizer, epoch, args)
        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'SB':
            train_sampler = None
            per_cls_weights = 1.0 / np.array(cls_num_list)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        criterion_sb = None
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        elif args.loss_type == 'SB':
            criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)
            criterion_sb = SBLoss(weight=per_cls_weights, alpha=1000).cuda(args.gpu)
        elif args.loss_type == 'EQL':
            criterion = EQL(cls_fre_per=cls_per).cuda(args.gpu)
        elif args.loss_type == 'CYC':
            criterion = Cyclical_FocalLoss().cuda(args.gpu)
        elif args.loss_type == 'LMF':
            criterion = LMFLoss(cls_num_list=cls_num_list,weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, criterion_sb, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, criterion_sb, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, criterion_sb, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if 'SB' in args.loss_type and epoch >= args.start_sb_epoch:
            output, features = model(input)
            loss = criterion_sb(output, target, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            output, features = model(input)
            loss = criterion_sb(output, target, features)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        else:
            output, _ = model(input)
            if 'CYC' in args.loss_type:
                loss=criterion(output, target,epoch)
            else:
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            output, _ = model(input)
            if 'CYC' in args.loss_type:
                loss = criterion(output, target, epoch)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        f1_macro =F1_macro(output, target)
        f1_micro = F1_macro(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        #f1.update(F1,input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, criterion_sb, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if 'SB' in args.loss_type and epoch >= args.start_sb_epoch:
                output, features = model(input)
                loss = criterion_sb(output, target, features)
                # Note that while the loss is computed using target, the target is not used for prediction.
            else:
                output, _ = model(input)
                if 'CYC' in args.loss_type:
                    loss = criterion(output, target, epoch)
                else:
                    loss = criterion(output, target)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            f1_micro = F1_micro(output, target)

            Gmean=AUC(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            #f1.update(F1, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        #print(f1_micro)
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f} F1_micro {F1_micro:.5f} G-Means {Gmean:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses,F1_micro=f1_micro,Gmean=Gmean))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch > args.start_sb_epoch and 'SB' in args.loss_type:
        lr = args.lr * 0.01
        if epoch > 180:
            lr = args.lr * 0.000001
        elif epoch > 160:
            lr = args.lr * 0.0001
    else:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch > 180:
            lr = args.lr * 0.0001
        elif epoch > 160:
            lr = args.lr * 0.01
        else:
            lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.rho_schedule == 'step':
      if epoch <= 5:
          rho = 0.05
      elif epoch > 180:
          rho = 0.6
      elif epoch > 160:
          rho = 0.5
      else:
          rho = 0.1
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
    if args.rho_schedule == 'linear':
      X = [1, args.epochs]
      Y = [args.min_rho, args.max_rho]
      y_interp = interp1d(X, Y)
      rho = y_interp(epoch)

      for param_group in optimizer.param_groups:

          param_group['rho'] = np.float16(rho)
    if args.rho_schedule == 'none':
      rho = args.rho
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho

if __name__ == '__main__':
    main()
