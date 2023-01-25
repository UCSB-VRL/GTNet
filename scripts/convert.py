'''
'''
mapper={

    'Conv_pretrain':'backbone',
    'Conv_people':'res_block_people.Conv',
    'Conv_objects':'res_block_obj.Conv',
    'Conv_context':'res_block_context.Conv',
    'spmap_up':'FC_S',
    'lin_embed_head':'FC_W.block',
    'lin_visual_head':'FC_B.block',
    'lin_single_tail':'FC_PB',
    'lin_trans_tail':'FC_P',
    'lin_single_head':'FC_PB_raw.block',
    'lin_trans_head':'lin_trans_head.block',


}
delete_keys={'lin_visual_tail'}


import argparse

import torch
from torch import nn

from copy import deepcopy
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, default='../soa_vcoco/bestcheckpoint.pth.tar'
    )
    parser.add_argument(
        '--save_path', type=str, default='../soa_vcoco/gtnet2bestcheckpoint.pth.tar'
    )


    args = parser.parse_args()

    return args


def main(args):
    old_model = torch.load(args.load_path)
    new_model={}
    for i in old_model:
        if i!='state_dict':
            new_model[i]=deepcopy(old_model[i])
        else:
            new_model['state_dict']={}
            for old_key in list(old_model[i].keys()):
                old_name=old_key.split('.')[1]
                if old_name in delete_keys:
                    continue
                elif old_name in mapper:
                    new_model['state_dict'][old_key.replace(old_name,mapper[old_name])]=old_model[i][old_key].clone()
                else:
                    new_model['state_dict'][old_key]=old_model[i][old_key].clone()
    import pdb;pdb.set_trace()

    torch.save(new_model, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)

