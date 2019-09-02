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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0

#@torchsnooper.snoop()
def main():
    global args, best_prec1
    args = parser.parse_args()

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
        model_1 = models.__dict__[args.arch]()
        num_ftrs = model_1.classifier[6].in_features
        model_1.classifier[6] = nn.Linear(num_ftrs, 3) #only train the last layer
        
        model_2 = models.__dict__[args.arch]()
        num_ftrs = model_2.classifier[6].in_features
        model_2.classifier[6] = nn.Linear(num_ftrs, 3) #only train the last layer
    
        model_3 = models.__dict__[args.arch]()
        num_ftrs = model_3.classifier[6].in_features
        model_3.classifier[6] = nn.Linear(num_ftrs, 3) #only train the last layer
    

    

    if args.gpu is not None:
        model_1 = model_1.cuda(args.gpu) #this way
        model_2 = model_2.cuda(args.gpu) #this way
        model_3 = model_3.cuda(args.gpu) #this way
    elif args.distributed:
        model_1.cuda()
        model_2.cuda()
        model_3.cuda()
        model_1 = torch.nn.parallel.DistributedDataParallel(model_1)
        model_2 = torch.nn.parallel.DistributedDataParallel(model_2)
        model_3 = torch.nn.parallel.DistributedDataParallel(model_3)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model_1.features = torch.nn.DataParallel(model_1.features)
            model_2.features = torch.nn.DataParallel(model_2.features)
            model_3.features = torch.nn.DataParallel(model_3.features)
            model_1.cuda()
            model_2.cuda()
            model_3.cuda()

        else:
            model = torch.nn.DataParallel(model).cuda()
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        re_path1 = os.path.join(args.resume, 'data1/scratch3c.pth.tar')
        re_path2 = os.path.join(args.resume, 'data2/scratch3c.pth.tar')
        re_path3 = os.path.join(args.resume, 'data3/scratch3c.pth.tar')
        assert os.path.isfile(re_path1), 'Error: no checkpoint1 directory found!'
        assert os.path.isfile(re_path2), 'Error: no checkpoint2 directory found!'
        assert os.path.isfile(re_path3), 'Error: no checkpoint3 directory found!'
        checkpoint1 = torch.load(re_path1).get('state_dict')
        checkpoint2 = torch.load(re_path2).get('state_dict')
        checkpoint3 = torch.load(re_path3).get('state_dict')
        model_1.load_state_dict(checkpoint1) #cat dog
        model_2.load_state_dict(checkpoint2) #cat rabbit
        model_3.load_state_dict(checkpoint3) #dog rabbit

    valdir_test = os.path.join(args.data, 'test/')

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

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)    
    test_acc = validate(test_loader, model_1, model_2, model_3, criterion)
   

    ''' a test
    model_new = models.__dict__[args.arch]()
    num_ftrs = model_new.classifier[6].in_features
    model_new.classifier[6] = nn.Linear(num_ftrs, 3)
    model_new = model_new.cuda(args.gpu)

    print("=> loading checkpoint ..")
    checkpoint = torch.load("/home/leander/hcc/prunWeight/save/pruned.pth.tar")
    #print(checkpoint['state_dict'])

    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace(".module", "") # removing ‘.moldule’ from key
        new_checkpoint[name]=v

    #print("new_checkpoint:",new_checkpoint)

    model_new.load_state_dict(new_checkpoint['state_dict'])
    validate(test_loader, model_new, criterion)
'''

    
    return

def validate(val_loader, model_1, model_2, model_3, criterion):
    # AverageMeter() : Computes and stores the average and current value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #zero_tensor = torch.FloatTensor(n, m)

    # switch to evaluate mode
    model_1.eval()
    model_2.eval()
    model_3.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True) # 0*100 + 1*100 +2*100
            print("target:",target)
            # compute output,out put is a tensor
            output_1 = model_1(input)
            output_1= F.softmax(output_1, dim=1) # calculate as row
            output_2 = model_2(input)
            output_2= F.softmax(output_2, dim=1)
            output_3 = model_3(input)
            output_3= F.softmax(output_3, dim=1)
            #print(output_2)

            out_size = output_1.size()
            row = out_size[0] 
            zero_tensor = torch.FloatTensor(row,1).zero_().cuda()

            o1_1 , o1_2 ,o1_3= output_1.chunk(3,dim=1)
            for x in range(row):
                o1_1[x][0] = o1_1[x][0]*o1_1[x][0]
                o1_2[x][0] = o1_2[x][0]*o1_2[x][0]
                o1_3[x][0] = o1_3[x][0]*o1_3[x][0]
            output_1 = torch.cat([o1_1,o1_2,zero_tensor],dim=1)

            o2_1 , o2_2, o2_3 = output_2.chunk(3,dim=1)
            for x in range(row):
                o2_1[x][0] = o2_1[x][0]*o2_1[x][0]
                o2_2[x][0] = o2_2[x][0]*o2_2[x][0]
                o2_3[x][0] = o2_3[x][0]*o2_3[x][0]
            output_2 = torch.cat([o2_1,zero_tensor,o2_3],dim=1)

            o3_1 , o3_2, o3_3 = output_3.chunk(3,dim=1)
            for x in range(row):
                o3_1[x][0] = o3_1[x][0]*o3_1[x][0]
                o3_2[x][0] = o3_2[x][0]*o3_2[x][0]
                o3_3[x][0] = o3_3[x][0]*o3_3[x][0]
            output_3 = torch.cat([zero_tensor, o3_2,o3_3],dim=1)

            #print(output_2)

            output = output_1 + output_2 + output_3
            print("output_1:",output_1)
            print("output_2:",output_2)
            print("output_3:",output_3)
            print("output:",output)

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