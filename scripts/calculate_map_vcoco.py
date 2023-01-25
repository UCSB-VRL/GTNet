import sys
import json

with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)	
path=all_data_dir+'v-coco/'
sys.path.insert(0, path)
import __init__

import vsrl_utils as vu
import numpy as np
import argparse
import pickle
parser=argparse.ArgumentParser()
parser.add_argument('-sa','--saving_epoch',type=int,required=False,default=5,help='In which epoch to save')
parser.add_argument('-fw','--first_word',type=str,required=False,default='result',help='Name_of_the_file_you_want_to_save')
parser.add_argument('-t','--types_of_data',type=str,required=False,default='train',help='Which_type_of_data')
args=parser.parse_args()
saving_epoch=args.saving_epoch
first_word=args.first_word
flag=args.types_of_data
folder_name='{}'.format(first_word)
from vsrl_eval import VCOCOeval
if flag=='train':
	vsrl_annot_file_s=path+'/data/vcoco/vcoco_train.json'
	split_file_s=path+'/data/splits/vcoco_train.ids'

elif flag=='test':
	vsrl_annot_file_s=path+'/data/vcoco/vcoco_test.json'
	split_file_s=path+'/data/splits/vcoco_test.ids'
elif flag=='val':
	vsrl_annot_file_s=path+'/data/vcoco/vcoco_val.json'
	split_file_s=path+'/data/splits/vcoco_val.ids'

coco_file_s=path+'/data/instances_vcoco_all_2014.json'
vcocoeval = VCOCOeval(vsrl_annot_file_s, coco_file_s, split_file_s)

file_name='../'+folder_name+'/'+'{}{}.pickle'.format(flag,saving_epoch)
print(file_name)
with open(file_name, 'rb') as handle:
    b = pickle.load(handle)
print(len(b))
vcocoeval._do_eval(b, ovr_thresh=0.5)

