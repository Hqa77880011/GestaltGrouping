import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,region,boundary,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24)
        self.encoder=Encoder(3,w);self.similarity=nn.Linear(w,w);self.value=nn.Linear(w,w)
        self.cue=nn.Linear(4,1);self.pair=nn.Linear(2*w,1);self.norm=nn.LayerNorm(w)
        self.gate=nn.Conv2d(2*w,w,1);self.head=nn.Conv2d(w,1,1);self.edge=nn.Conv2d(w,1,1)
    def grouped(self,h,removed=None):
        size=self.c.get('grid',6);token=F.adaptive_avg_pool2d(h,(size,size)).flatten(2).transpose(1,2)
        yy,xx=torch.meshgrid(torch.linspace(0,1,size,device=h.device),torch.linspace(0,1,size,device=h.device),indexing='ij')
        pos=torch.stack([xx.flatten(),yy.flatten()],-1);distance=torch.cdist(pos,pos).square()
        similarity=torch.exp(-torch.cdist(self.similarity(token),self.similarity(token)).square()/token.shape[-1])
        edge=F.adaptive_avg_pool2d(self.edge(h).sigmoid(),(size,size))
        dx=F.pad(edge[:,:,:,1:]-edge[:,:,:,:-1],(0,1,0,0));dy=F.pad(edge[:,:,1:]-edge[:,:,:-1],(0,0,0,1))
        tangent=F.normalize(torch.stack([-dy.flatten(1),dx.flatten(1)],-1),dim=-1)
        continuity=torch.exp(-(1-abs(tangent@tangent.transpose(1,2)))/.3)
        closure=torch.exp(-distance/.1)[None]*torch.sqrt(edge.flatten(1)[:,:,None]*edge.flatten(1)[:,None,:]+1e-6)
        cues=torch.stack([torch.exp(-distance/.2)[None].expand_as(similarity),similarity,continuity,closure],-1)
        names=['no_proximity','no_similarity','no_continuity','no_closure']
        for index,name in enumerate(names):
            if self.c['variant']==name or removed==index:cues=cues*torch.tensor([j!=index for j in range(4)],device=h.device)
        pair=torch.cat([token[:,:,None].expand(-1,-1,token.shape[1],-1),token[:,None].expand(-1,token.shape[1],-1,-1)],-1)
        attention=(self.pair(pair)+self.cue(cues)).tanh().squeeze(-1).softmax(-1)
        grouped=self.norm(token+attention@self.value(token)).transpose(1,2).reshape(h.shape[0],h.shape[1],size,size)
        grouped=F.interpolate(grouped,h.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(h+self.gate(torch.cat([h,grouped],1)).sigmoid()*grouped).sigmoid(),attention
    def forward(self,b):
        h=self.encoder(b['x']);p,attention=self.grouped(h)
        if self.c['variant']=='baseline':p=self.head(h).sigmoid()
        result={'probability':p,'attention':attention}
        if self.training and self.c['variant']=='full':result['interventions']=torch.stack([self.grouped(h,i)[0] for i in range(4)],1)
        return result
    def objective(self,b,p):
        loss=region(p['probability'],b['y'])+.2*F.l1_loss(boundary(p['probability'],1),boundary(b['y'],1))
        if 'interventions' in p:
            loss=loss+.1*torch.stack([F.relu(self.c.get('margin',.02)+loss-region(p['interventions'][:,i],b['y'])) for i in range(4)]).mean()
        return loss


def demo(seed,n,c):return image_demo(seed,n)
