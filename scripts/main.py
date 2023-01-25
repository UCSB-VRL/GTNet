from __future__ import print_function, division
import torch
import torch.nn as nn
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim 
import warnings
warnings.filterwarnings("ignore")  

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader_vcoco import Rescale,ToTensor,vcoco_Dataset,vcoco_collate
from train_test import train_test
import random
from gtnet import GTNet
device = torch.device("cuda")


seed=10
#seed=25000
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))


##### Arguments #######
parser=argparse.ArgumentParser()
parser.add_argument('-e','--number_of_epochs',type=int,required=False,default=100,help='Number of epochs to run the model')
parser.add_argument('-l','--learning_rate',type=float,required=False,default=0.001,help='Initial learning_rate')
parser.add_argument('-b','--breaking_point',type=int,required=False,default=0,help='Number of iteration to break the training loop')
parser.add_argument('-sa','--saving_epoch',type=int,required=False,default=10,help='In which epoch to save')
parser.add_argument('-fw','--first_word',type=str,required=False,default='result',help='Name of the folder in which you want to save')
parser.add_argument('-ba','--batch_size',type=int,required=False,default=8,help='Batch size')
parser.add_argument('-r','--resume_model',action='store_true',help='Flag for pretrained model loading')
parser.add_argument('-i','--inference',action='store_true',help='Flag for running inference')
parser.add_argument('-h_l','--hyper_load',type=str,required=False,default='f',help='If this flag is t then the model will load stored hyper parameters')
parser.add_argument('-v_i','--Visualize',type=str,required=False,default='f',help='Mention which phase(train,val,test) you want to visualize')
parser.add_argument('-c','--check_point',type=str,required=False,default='best',help='First word of the checkpoint file which would be followed by checkpoint.pth.tar')


args=parser.parse_args()

number_of_epochs=args.number_of_epochs
learning_rate=args.learning_rate
breaking_point=args.breaking_point
saving_epoch=args.saving_epoch
first_word=args.first_word
batch_size=args.batch_size
resume_model=args.resume_model
infr=args.inference
hyp=args.hyper_load
visualize=args.Visualize
check=args.check_point
###############################################################################3

with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)

annotation_train=all_data_dir+'Annotations_vcoco/train_annotations.json'
image_dir_train=all_data_dir+'Data_vcoco/train2014/'

annotation_val=all_data_dir+'Annotations_vcoco/val_annotations.json'
image_dir_val=all_data_dir+'Data_vcoco/train2014/'

annotation_test=all_data_dir+'Annotations_vcoco/test_annotations.json'
image_dir_test=all_data_dir+'Data_vcoco/val2014/'



vcoco_train=vcoco_Dataset(annotation_train,image_dir_train,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))
vcoco_val=vcoco_Dataset(annotation_val,image_dir_val,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))
vcoco_test=vcoco_Dataset(annotation_test,image_dir_test,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))


dataloader_train = DataLoader(vcoco_train, batch_size,
                        shuffle=True,collate_fn=vcoco_collate,num_workers=4,worker_init_fn=_init_fn)
dataloader_val = DataLoader(vcoco_val, batch_size,
                        shuffle=True,collate_fn=vcoco_collate,num_workers=4,worker_init_fn=_init_fn)
dataloader_test = DataLoader(vcoco_test, batch_size,
                        shuffle=False,collate_fn=vcoco_collate,num_workers=4, worker_init_fn=_init_fn)
dataloader={'train':dataloader_train,'val':dataloader_val,'test':dataloader_test}



folder_name='../{}'.format(first_word)



### Loading Model ###
res=GTNet().float()
frozen_points=['Conv_pretrain.0','Conv_pretrain.1','Conv_pretrain.2','Conv_pretrain.3','Conv_pretrain.4','Conv_pretrain.5']
trainables=[]
not_trainables=[]
pretrain_tune=[]
spmap=[]
single=[]
for name, p in res.named_parameters():
    if name.split('.')[0]=='Conv_pretrain':
        p.requires_grad=False
        not_trainables.append(p)
    else:
        if name.split('.')[0]=='conv_sp_map' or name.split('.')[0]=='spmap_up':
            spmap.append(p)
        else:
            trainables.append(p)
	
optim1 = optim.SGD([
                    {"params":trainables,"lr":learning_rate},
                    {"params":spmap,"lr":learning_rate},
                    ],
                    momentum=0.9,weight_decay=0.0001)

lambda1 = lambda epoch: 1.0 if epoch < 9 else (10 if epoch < 32 else 1) 
scheduler=optim.lr_scheduler.LambdaLR(optim1,[lambda1,lambda1])
res=nn.DataParallel(res)
res.to(device)
epoch=0
mean_best=0
if resume_model:
        try:
            checkpoint=torch.load(folder_name+'/'+check+'checkpoint.pth.tar',encoding='latin1')
            
            res.load_state_dict(checkpoint['state_dict'],strict=True)

            epoch = checkpoint['epoch']
            mean_best=checkpoint['mean_best']
            print("=> loaded checkpoint when best_prediction {} and epoch {}".format(mean_best, checkpoint['epoch']))
        except:
            print('Failed to load checkPoint')
if hyp=='t':
    try:
        print('Loading previous Hyperparameters')
        optim1.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
         print('Failed to load previous Hyperparameters')
train_test(res,optim1,scheduler,dataloader,number_of_epochs,breaking_point,saving_epoch,folder_name,batch_size,infr,epoch,mean_best,visualize)


