import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable

# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout = 0.3):
        super(FeedForward, self).__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
      #  nn.init.normal(self.linear_1.weight, std=0.001)  
      #  nn.init.normal(self.linear_2.weight, std=0.001)  

    def forward(self, x):
        #x = self.dropout(F.relu(self.linear_1(x)))
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return F.relu(x)

# standard NORM layer of Transformer
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def add_neighbour(scores_c,int_prob,num_peo,num_obj):
    #import pdb;pdb.set_trace()    
    pa,h,w=scores_c.shape
    #scores_c=scores.detach().clone()
    add=torch.zeros(scores_c.shape).cuda().float()
    weighted_scores=(int_prob*scores_c.view(pa,h*w)).view_as(scores_c)
    sum_=weighted_scores.reshape(num_peo,num_obj,h,w).sum(1).reshape(num_peo,1,h,w)   
    add2=(sum_-weighted_scores.reshape(num_peo,num_obj,h,w)).reshape(num_peo*num_obj,h,w)
    

   # start_c=0
   # for peo in range(num_peo):
   #     detached_scores_=weighted_scores[start_c:start_c+num_peo*num_obj]
   #     detached_scores=torch.sum(detached_scores_,0)
   #     for obj in range(num_obj):
   #         #pass
   #         add[start_c+peo+obj]=detached_scores-(scores_c[start_c+peo+obj]*int_prob[start_c+peo+obj])
   #         #import pdb;pdb.set_trace()    
   #     start_c=start_c+num_peo*num_obj 
    scores_f=scores_c+add2
    #assert add==add2
    #import pdb;pdb.set_trace()    
    return scores_f
# standard attenccon layer
def attention(q, k, v, d_k, mask=None, dropout=None):
    pa,c,h,w=q.shape[0],q.shape[1],k.shape[-1],k.shape[-2]
    
    #torch.autograd.set_detect_anomaly(True)
    scores=(torch.mm(q,k[0].reshape(c,h*w)).reshape(pa,h,w))/ math.sqrt(d_k)    
    #import pdb;pdb.set_trace()
    #up_scores=add_neighbour(scores[0],stat[1])
    scores = nn.Softmax(-1)(scores.view(pa,-1 )).view_as(scores)
    # scores : b, t 
    #import pdb;pdb.set_trace()
    #y_up=add_neighbour(y[0],stat[1])
    #y_up = nn.Softmax(-1)(up_scores.view(pa,-1 )).view_as(up_scores)
    
    
    output=((scores.unsqueeze(1))*v).sum(-1).sum(-1)
    #output=((y_up.unsqueeze(1))*v).sum(-1).sum(-1)
    if dropout:
        output = dropout(output)
    return output




class TX(nn.Module):
    def __init__(self, d_model=256 , dropout = 0.3 ):
        super(TX, self).__init__()
        self.d_model = d_model
        # no of head has been modified to encompass : 1024 dimension 
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=d_model/2)
    def forward(self, q, k, v, mask=None):
        # q: (b , dim )

        A = attention(q, k, v, self.d_model,mask, self.dropout)
        # A : (b , d_model=1024 // 16 )
        #import pdb;pdb.set_trace()
        try:
            q_ = self.norm_1(A + q)
            new_query = self.norm_2(q_ +  self.dropout_2(self.ff(q_))) 
        except:
            import pdb;pdb.set_trace()
        #new_query = self.ff(A)*q+q
        return new_query


class k_v_projs(nn.Module):
    def __init__(self,in_f=1024, d_tot=512 ):
    #def __init__(self,in_f=2048, d_tot=512 ):
        super(k_v_projs, self).__init__()
        self.Conv_k=nn.Sequential(nn.Conv2d(in_f, d_tot, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(d_tot, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False),
				)
        self.Conv_v=nn.Sequential(nn.Conv2d(in_f, d_tot, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(d_tot, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False),
				)
    def forward(self,f_map):
        
        return self.Conv_k(f_map),self.Conv_v(f_map)

class Block_head(nn.Module):
    def __init__(self, d_model=256,dropout = 0.3 ):
        super(Block_head, self).__init__()
        self.T1 = TX(d_model)
       # self.T2 = TX()
       # self.T3 = TX()
    def forward(self, q, k, v, mask=None):
        q = self.T1(q,k,v)
       # q = self.T2(q,k,v)
       # q = self.T3(q,k,v)
        return q

class combining_head(nn.Module):
    def __init__(self, d_tot=512,dropout = 0.3 ):
        super(combining_head, self).__init__()
        self.linear=nn.Linear(d_tot,d_tot)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.linear(x))
        
        return x

class Action_Transformer(nn.Module):
    def __init__(self,d_tot=512, head=2,layers=2):
        super(Action_Transformer, self).__init__()
        
        self.head=head
        self.head_layers =[]
        self.combining_layers =[]
        self.k_v_projs =k_v_projs(in_f=d_tot*2,d_tot=d_tot)
        self.d_model=int(d_tot/self.head)
        self.layers=layers
        for i in range(self.head*self.layers):
            self.head_layers.append(Block_head(self.d_model))
        for i in range(self.layers):
            self.combining_layers.append(combining_head(d_tot))

        
        self.list_layers = nn.ModuleList(self.head_layers)
        self.list_combining_layers = nn.ModuleList(self.combining_layers)
        #import pdb;pdb.set_trace()
        
    def forward(self,f_map,q,att_g):
        k,v=self.k_v_projs(f_map)
        #import pdb;pdb.set_trace()
        #v=v*ch_att.unsqueeze(-1).unsqueeze(-1)
        #start=0
        flag=0
        #print('new') 
        for j in range(self.layers):
            start=0 
            outputs = []
            for i in range(self.head):
                #import pdb;pdb.set_trace() 
                k_head=k[:,start:start+self.d_model,:,:]
                v_head=v[:,start:start+self.d_model,:,:]
                q_head=q[:,start:start+self.d_model]
                outputs.append(self.list_layers[flag](q_head,k_head, v_head) )
                start+=self.d_model
                flag+=1
            
            f = torch.cat(outputs, 1)
            q=F.relu(self.list_combining_layers[j](f))*att_g
            #q=F.relu(self.list_combining_layers[j](f))
            #q=self.list_combining_layers[j](f)
            #import pdb;pdb.set_trace()
             
        #q = F.normalize(q, p=2, dim=1)
        # F.norma
        return q


        
if __name__=='__main__':
    f_map=torch.rand(1,1024,25,25).cuda(1)
    v=torch.rand(1,1024,25,25).cuda(1)
    q=torch.rand(12,512).cuda(1)
    num_classes=29

    model=Action_Transformer().cuda(1)
    model(f_map,q)
