from __future__ import print_function, division
import json
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import helpers_preprocess as labels
import utils_pre as labels
from PIL import Image
import matplotlib.pyplot as plt
bad_detections_train,bad_detections_val,bad_detections_test=labels.dry_run()
#bad_detections_train,bad_detections_val,bad_detections_test=[],[],[]
NO_VERB=29
def vcoco_collate(batch):
    image =[] 
    image_id=[]
    pairs_info=[]
    labels_all=[]
    labels_single=[]
    all_utils={}
    for index,item in enumerate(batch):
        image.append(item['image'])
        image_id.append(torch.tensor(int(item['image_id'])))
        pairs_info.append(torch.tensor(item['pairs_info']))
        labels_all.append(torch.tensor(item['labels_all']))
        labels_single.append(torch.tensor(item['labels_single']))
        all_utils[int(item['image_id']),item['flag']]=item['all_utils']
    
    return [torch.stack(image),torch.cat(labels_all),torch.cat(labels_single),torch.stack(image_id),torch.stack(pairs_info),all_utils]

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self,all_inn):
        image=all_inn


        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        #import pdb;pdb.set_trace()
        img2 = transform.resize(image, (new_h, new_w))
        
        return img2



def augmentation(image,boxes):

    image = image.transpose((2, 0, 1))
       
    return torch.from_numpy(image).float()
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, all_inn):
        image=all_inn

        #import pdb;pdb.set_trace()
        	
        image = image.transpose((2, 0, 1))
       
        return torch.from_numpy(image).float()


class vcoco_Dataset(Dataset):
    

    def __init__(self, json_file_image,root_dir,transform=None):
        with open(json_file_image) as json_file_:               
            self.vcoco_frame_file = json.load(json_file_)
        self.flag=json_file_image.split('/')[-1].split('_')[0]
        if self.flag=='train':
            self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_train)]  
        elif self.flag=='val':
            self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_val)]	
        elif self.flag=='test':
            self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_test)]
            self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.vcoco_frame)

    def __getitem__(self, idx):
        if self.flag=='test':
            img_pre_suffix='COCO_val2014_'+str(self.vcoco_frame[idx]).zfill(12)+'.jpg'
        else:
            img_pre_suffix='COCO_train2014_'+str(self.vcoco_frame[idx]).zfill(12)+'.jpg'
        all_labels=labels.get_detections(int(self.vcoco_frame[idx]),self.flag)
        labels_all=all_labels['labels_all']
        labels_single=all_labels['labels_single']
        image=all_labels['image']
        num_pers=len(all_labels['person_bbx'])
        num_objs=len(all_labels['objects_bbx'])
        num_hois=len(labels_single)
            
        del all_labels['image']
        if self.transform:
            image = self.transform(image)
        sample = {'image':image ,'labels_all':labels_all,'labels_single':labels_single,'image_id':self.vcoco_frame[idx],'all_utils':all_labels,'flag':self.flag,'pairs_info':[num_pers,num_objs,num_hois]}
        return sample
