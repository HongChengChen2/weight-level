import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torchsnooper
import torchsummary

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict


#63.333 63.333

#only used data to compute accuracy, not in deciding which to prune

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--percent',default=0.1,type=float)

best_prec1 = 0

#@torchsnooper.snoop()
def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    if args.gpu is not None:
        model = model.cuda(args.gpu) #this way
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()

        else:
            model = torch.nn.DataParallel(model).cuda()


    valdir_test = os.path.join(args.data, 'test/')
    valdir_val = os.path.join(args.data, 'val/')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    test_dataset = datasets.ImageFolder(valdir_test, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir_val, transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset , batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    
    for param in model.parameters(): #params have requires_grad=True by default
        param.requires_grad = False #only train the last layer:fc layer
        param.cuda(args.gpu)
    

    #print("architecture of alexnet:")
    #print(model)

    optimizer = optim.Adam(model.parameters(),lr=0.001)


    test_acc0 = validate(test_loader, model, criterion)
    print("==========================================")
    test_acc0 = validate(val_loader, model, criterion)
    
    return

def validate(val_loader, model, criterion):
    # AverageMeter() : Computes and stores the average and current value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True) # 0*100 + 1*100 +2*100
            #print("target:",target)
            # compute output,out put is a tensor
            output = model(input)

            print("output:",output)
            #print("[0][0] :",output[0][0].item())

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #view() means resize() -1 means 'it depends'
    with torch.no_grad():
        batch_size = target.size(0)
        #print("batch_size",batch_size)
        maxk = max(topk) # = 5
        _, pred = output.topk(maxk, 1, True, True) #sort and get top k and their index
        #print("pred:",pred) #is index 5col xrow

        for x in range(0,pred.size()[0]):
            for y in range(0,pred.size()[1]):
                if pred[x][y] >=281 and pred[x][y]<=285 : #cats
                    pred[x][y] = 0
                    break
                elif pred[x][y] >=151 and pred[x][y]<=268 : #dogs
                    pred[x][y] = 1
                    break               
                elif pred[x][y] >=330 and pred[x][y]<=332 : #rabbits
                    pred[x][y] = 2
                    break
                else:
                    pred[x][y] = 3
        #print("pred after:",pred)

        pred = pred.t() # a zhuanzhi transpose xcol 5row
        #print("pred.t():",pred)
        #print("size:",pred[0][0].type()) #5,12


        correct = pred.eq(target.view(1, -1).expand_as(pred)) #expend target to pred

        res = []
        for k in topk: #loop twice 1&5 
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()