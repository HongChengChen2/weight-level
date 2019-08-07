import argparse
import os
import random
import shutil
import warnings
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt

from collections import OrderedDict
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
parser.add_argument('--save',default='',type=str)

#python show.py --arch vgg16_bn --resume /Users/hongchengchen/develop/wl_store/ShowCnn/vgg16/data1/scratch.pth.tar 

def createModel():
    global args, best_prec1
    args = parser.parse_args()

    #load model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)  #resNet50

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume,map_location ='cpu').get('state_dict')
        #print(checkpoint.keys())
        
        new_checkpoint = OrderedDict()

        
        for k, v in checkpoint.items():
            name = k.replace(".module", "") # removing ‘.moldule’ from key
            new_checkpoint[name]=v
            #print("new_checkpoint:",new_checkpoint)
        '''
        for k, v in checkpoint.items():
            if 'classifier' in k :
                new_checkpoint[k]=v

            elif 'module' not in k :
                k = 'module.'+k
                #k = k.replace('module.features.','features.module.')
                #print("new_k", k)
                new_checkpoint[k]=v
        '''
        model.load_state_dict(new_checkpoint)

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    
    return model

def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


if __name__ == '__main__':
    '''   
     # The local path to our target image
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    preds = model.predict(x)
    '''
    img = cv2.imread('000.jpg')
 
    model = createModel()
 
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model(torch.from_numpy(img_batch))  # conv_img 卷积结果
 
    visualize_feature_map(conv_img)
