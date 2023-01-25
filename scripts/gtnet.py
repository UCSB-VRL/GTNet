from __future__ import print_function, division
import torch
import torch.nn as nn

import os
import numpy as np
import pool_pairing  as ROI 
import random
import torchvision.models as models
from guided_transformer import *


class Flatten(nn.Module):
   def __init__(self):
        super(Flatten,self).__init__()
 
   def forward(self, x):
        return x.view(x.size()[0], -1)

class GTNet_resblock(nn.Module):

    def __init__(self,in_out_dim=1024,hid_dim=512):
        super(GTNet_resblock,self).__init__()
        self.Conv=nn.Sequential(
                    nn.Conv2d(in_out_dim, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(hid_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(hid_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(hid_dim, in_out_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(in_out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=False),
                    )    

    def forward(self,x):
        return self.Conv(x)+x

class FC_block(nn.Module):
        
        def __init__(self,in_dim=1024,hid_dim=1024,out_dim=512):
                super(FC_block,self).__init__()
                self.block=nn.Sequential(
	            nn.Linear(in_dim, hid_dim),
                    nn.Linear(hid_dim,out_dim),
                    nn.ReLU(),

                )
        def forward(self,x):
                return self.block(x)

class GTNet(nn.Module):
        def __init__(self,lin_size=1024,embedding_vector_size=600,pool_size=(10,10),out_size=512):
                super(GTNet,self).__init__()
                self.flat= Flatten()
                model =models.resnet152(pretrained=True)
                self.backbone = nn.Sequential(*list(model.children())[0:7])## Resnets,resnext                
                
                ######### Residual Blocks for human,objects and the context##############################
                self.res_block_people=GTNet_resblock()
                self.res_block_obj=GTNet_resblock()
                self.res_block_context=GTNet_resblock()	
 
                self.transformer= GTNet_Transformer()
                self.conv_sp_map=nn.Sequential(
                                nn.Conv2d(2, 64, kernel_size=(5, 5)),
                                #nn.Conv2d(3, 64, kernel_size=(5, 5)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                                nn.Conv2d(64, 32, kernel_size=(5,5)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                        	#nn.AvgPool2d((13,13),padding=0,stride=(1,1)),
                        	nn.AdaptiveAvgPool2d((1,1)),

				    
			     )
                self.FC_S=nn.Sequential(nn.Linear(32,512),nn.ReLU(),)

		#Prediction model for embedding vectors######################
                self.FC_W=FC_block(in_dim=embedding_vector_size,hid_dim=embedding_vector_size,out_dim=out_size)

		#Interaction prediction model for visual features######################
                self.FC_PB_raw=FC_block(in_dim=lin_size*3+4,hid_dim=lin_size,out_dim=out_size)
                self.FC_PB=nn.Sequential(nn.Linear(out_size*2,1),)
                ########## Prediction model for visual features#################
                self.FC_B=FC_block(in_dim=lin_size*3+4,hid_dim=lin_size,out_dim=out_size)
                ################################################
                self.sigmoid=nn.Sigmoid()
                self.relu=nn.ReLU()
                ####### Prediction model for transformer features##################
                self.lin_trans_head=FC_block(in_dim=lin_size*2+4,hid_dim=lin_size,out_dim=out_size)
                self.FC_P=nn.Sequential(nn.Linear(out_size*3,29),)
                self.pool_size=pool_size
                ########################################	
        def forward(self,x,pairs_info,pairs_info_augmented,image_id,flag_,phase):
                F = self.backbone(x)###        
                rois_people,rois_objects,spatial_locs,union_box,objects_embed,stats= ROI.get_pool_loc(F,image_id,flag_,size=self.pool_size,spatial_scale=25,batch_size=len(pairs_info))
                ### Defining The Pooling Operations ####### 
                x,y=F.size()[2],F.size()[3]	
                hum_pool=nn.AvgPool2d(self.pool_size,padding=0,stride=(1,1))
                obj_pool=nn.AvgPool2d(self.pool_size,padding=0,stride=(1,1))
                context_pool=nn.AvgPool2d((x,y),padding=0,stride=(1,1))
                #################################################
                
                ### Human###
                res_people=self.res_block_people(rois_people)
                res_av_people=hum_pool(res_people)
                f_H=self.flat(res_av_people)
                ##Objects##
                res_objects=self.res_block_obj(rois_objects)
                res_av_objects=obj_pool(res_objects)
                f_O=self.flat(res_av_objects)

                #### Context ######
                res_context=self.res_block_context(F)
                res_av_context=context_pool(res_context)
                f_G=self.flat(res_av_context)

                f_s=self.FC_S(self.flat(self.conv_sp_map(union_box)))
                #### Prediction from embedding vectors####
                f_w=self.FC_W(objects_embed)
                ## Generating guidance ##
                guidance=f_w*f_s	
                #### Making Essential Pairing##########
                #pairs,people,objects_only,pos_embed= ROI.pairing(f_H,f_O,f_G,spatial_locs,pairs_info)
                pairs,pairs_only= ROI.pairing(f_H,f_O,f_G,spatial_locs,pairs_info,stats)
                ###### Interaction Probability##########
                b_I_raw=self.FC_PB_raw(pairs)
                b_I_guided=b_I_raw*guidance
                b_I=self.FC_PB(torch.cat((b_I_raw,b_I_guided),1))	
                f_Q=self.lin_trans_head(pairs_only)
                f_GQ=f_Q*guidance
                start_c=0
                f_C_list=[] 
                #### Prediction from visual features####
                f_B=self.FC_B(pairs)
                f_BR=f_B*guidance
                for batch_num,l in enumerate(pairs_info):
                        incre=int(len(stats[batch_num][2]))
                        f_GQ_batch=f_GQ[start_c:start_c+incre]  
                        guidance_batch=guidance[start_c:start_c+incre]  
                        f_C_list.append(self.transformer(F[batch_num].unsqueeze(0),f_GQ_batch,guidance_batch))

                        ###Loop increment for next batch##
                        start_c+=incre


                f_C=torch.cat(f_C_list)

                p_I=self.FC_P(torch.cat((f_B,f_BR,f_C),1))
                return [p_I,b_I]
