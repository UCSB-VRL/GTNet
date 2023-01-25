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
    def forward(self, x):
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

# standard attention layer
def attention(q, k, v, d_k, mask=None, dropout=None):
    pa,c,h,w=q.shape[0],q.shape[1],k.shape[-1],k.shape[-2]
    
    scores=(torch.mm(q,k[0].reshape(c,h*w)).reshape(pa,h,w))/ math.sqrt(d_k)    
    scores = nn.Softmax(-1)(scores.view(pa,-1 )).view_as(scores)    
    output=((scores.unsqueeze(1))*v).sum(-1).sum(-1)
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
        self.ff = FeedForward(d_model, d_ff=int(d_model/2))
    def forward(self, q, k, v, mask=None):
        A = attention(q, k, v, self.d_model,mask, self.dropout)
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ +  self.dropout_2(self.ff(q_))) 

        return new_query


class k_v_projs(nn.Module):
    def __init__(self,in_f=1024, d_tot=512 ):
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
    def forward(self, q, k, v, mask=None):
        q = self.T1(q,k,v)
        return q

class combining_head(nn.Module):
    def __init__(self, d_tot=512,dropout = 0.3 ):
        super(combining_head, self).__init__()
        self.linear=nn.Linear(d_tot,d_tot)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.linear(x))
        
        return x

class GTNet_Transformer(nn.Module):
    def __init__(self,d_tot=512, head=2,layers=2):
        super(GTNet_Transformer, self).__init__()
        
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
        
    def forward(self,f_map,q,att_g):
        k,v=self.k_v_projs(f_map)
        flag=0
        for j in range(self.layers):
            start=0 
            outputs = []
            for i in range(self.head):
                k_head=k[:,start:start+self.d_model,:,:]
                v_head=v[:,start:start+self.d_model,:,:]
                q_head=q[:,start:start+self.d_model]
                outputs.append(self.list_layers[flag](q_head,k_head, v_head) )
                start+=self.d_model
                flag+=1
            
            f = torch.cat(outputs, 1)
            q=F.relu(self.list_combining_layers[j](f))*att_g
        return q


        
if __name__=='__main__':
    f_map=torch.rand(1,1024,25,25).cuda(1)
    v=torch.rand(1,1024,25,25).cuda(1)
    q=torch.rand(12,512).cuda(1)
    num_classes=29

    model=GTNet_Transformer().cuda(1)
    model(f_map,q)
