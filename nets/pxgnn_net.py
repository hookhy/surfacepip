import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import dgl

"""
Unofficial implementaion of PxGNN
    Enyan Dai, and Suhang Wang, Towards Prototype-Based Self-Explainable Graph Neural
Network
    https://arxiv.org/abs/2210.01974
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class PxGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        # layer parameters
        self.device = net_params['device']
        in_dim = net_params['in_dim'] 
        hidden_dim = net_params['hidden_dim'] # D
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        
        # GNN encoder
        dropout = net_params['dropout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        n_layers = net_params['L']
        aggregator_type = net_params['sage_aggregator']
        
        # prototype layer
        self.num_prot_per_class = net_params['num_prot_per_class'] # K
        num_prototypes = int(self.num_prot_per_class*2)
        
        # 
        self.lambda_reg = net_params['lambda_reg']
        self.lambda_recon = net_params['lambda_recon']
        
        # initial embeddings
        # there are init graphs ([K] list of initial prototype DGL-format graphs)
        # self.g_pos_init [g_pos1, g_pos2, ..., g_posk]
        # self.g_neg_init [g_neg1, g_neg2, ..., g_negk]
        #
        # ...and there are also [K] list of initial prototype embeddings
        # self.h_pos_init [(N,D), (N,D), ..., (N,D)]
        # self.h_neg_init [(N,D), (N,D), ..., (N,D)]

        # initial embedding layers
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # GNN embedding layers
        self.encoders = nn.ModuleList([
            GraphSageLayer(hidden_dim, hidden_dim, F.relu, dropout, 
                           aggregator_type, self.batch_norm, self.residual) for _ in range(n_layers)]) 
        
        # prototype generator
        self.dec1 = MLPReadout(hidden_dim,in_dim)
        self.dec2 = nn.Linear(2*in_dim,1)

        # FC layers
        self.FC_layers = nn.Linear(num_prototypes,n_classes,bias=False)
 
    # get similarity value: hgi - size of (batch,dim), prototype - size of (dim)
    def get_sims_prototypes(self, hgi, prototype):
        distance = torch.norm(hgi - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + 1e-12))
        
        return similarity # size of (batch)
    
    def prot_generate(self, emb, g_init=None):
        x = self.dec1(emb) # N,D
        x_l = x.unsqueeze(1).repeat(1,x.size(0),1) # N,N,D
        x_r = x.unsqueeze(0).repeat(x.size(0),1,1) # N,N,D
        x_cat = self.dec2(torch.cat((x_l,x_r),dim=2)) # N,N,2*D
        
        s =  torch.sigmoid(x_cat.squeeze()) # N,N
        
        if g_init is not None:
            a_p = g_init.adjacency_matrix(ctx=self.device).to_dense() # N,N
            th_ = a_p * 0.2 + (1-a_p) * 0.8
            a = (s > th_).float()
            gn = dgl.graph((torch.nonzero(a)[:,0], torch.nonzero(a)[:,1]),num_nodes=a.size(0))
            gn.ndata['feat'] = x
            return gn
        
        else:
            return s, x
        
    def forward(self, g, h, e):
            
        # feature initial embedding: input_dim -> hidden_dim
        h = self.embedding_h(h)
     
        # GNN embedding: hidden_dim -> hidden_dim
        for conv in self.encoders:
            h = conv(g, h)   
        g.ndata['h'] =  h   
        self.hg = dgl.mean_nodes(g, 'h') 
        
        # reconstruction
        self.recon_hs = []
        self.recon_ss = []
        self.ind_graphs = dgl.unbatch(g)
        for one_g in self.ind_graphs:
            s, x = self.prot_generate(one_g.ndata['h'])
            self.recon_hs.append(x)
            self.recon_ss.append(s)
        
        # prototype generation
        pgs = []
        for prot in range(self.num_prot_per_class):
            pgs.append(self.prot_generate(self.p_neg[prot],self.g_neg_init[prot]))
        for prot in range(self.num_prot_per_class):
            pgs.append(self.prot_generate(self.p_pos[prot],self.g_pos_init[prot]))
            
        # prototype embedding
        hgs = []
        for pg in pgs:
            h = pg.ndata['feat']
            h = self.embedding_h(h)
            for conv in self.encoders:
                h = conv(pg, h)   
            pg.ndata['h'] =  h   
            hgs.append(dgl.mean_nodes(pg, 'h'))

        # prototype projection    
        S_ = []
        for one_prot_hg in hgs:
            S_.append(self.get_sims_prototypes(self.hg, one_prot_hg))
        
        self.ss = torch.cat(S_,dim=1) # (batch,2*K)
        weight_ss = torch.softmax(self.ss,dim=1)
        
        output_prob = torch.zeros(weight_ss.size(0),2).to(self.device)
        for p in range(weight_ss.size(1)):
            one_hot = torch.zeros_like(output_prob).to(self.device)
            one_hot[:,p//self.num_prot_per_class] = 1
            output_prob +=  one_hot * weight_ss[:,p][:,None]

        return output_prob[:,1] # (batch)
    
    def loss(self, pred, label):
        
        nb = label.size(0) # number of batch
        
        # supervised loss
        #criterion = nn.BCELoss()
        #sup_loss = criterion(pred, label)
        exp_sims = torch.exp(self.ss) # (batch,2*K)
        denom = torch.sum(exp_sims,dim=1) # batch
        mask = torch.zeros_like(exp_sims)
        for i in range(nb):
            mask[i,int(label[i])*self.num_prot_per_class:int(label[i])*self.num_prot_per_class+self.num_prot_per_class] = 1
        numer = torch.sum(exp_sims * mask,dim=1) # batch
        sup_loss = torch.mean(-torch.log( numer / (denom + 1e-12) ))

        # regularization
        loss_reg = 0
        for prot in range(self.num_prot_per_class):
            loss_reg += torch.norm(self.p_pos[prot]-self.h_pos_init[prot],p='fro')
            loss_reg += torch.norm(self.p_neg[prot]-self.h_neg_init[prot],p='fro')
        loss_reg /= 2
        loss_reg /= self.num_prot_per_class
        
        # recon loss
        loss_recon = 0
        for idx, one_g in enumerate(self.ind_graphs):
            one_adj = one_g.adjacency_matrix(ctx=self.device).to_dense()
            
            # for negative sampling (Q=1000)
            r,c = torch.where(one_adj==0)
            neg_samp_idx = torch.randperm(r.numel())[:r.numel()-1000]
            rs, cs = r[neg_samp_idx], c[neg_samp_idx]
            adj_negfilled = one_adj.clone()
            adj_negfilled[rs,cs] = 1
            
            loss_recon += torch.norm(one_g.ndata['feat']-self.recon_hs[idx],p='fro')
            loss_recon += torch.mean(
                -one_adj * torch.log(self.recon_ss[idx])
                - (1-adj_negfilled) * torch.log(1-self.recon_ss[idx])
            )
        loss_recon /= nb
        
        total_loss = ( 
                        sup_loss 
                      + self.lambda_reg * loss_reg 
                      + self.lambda_recon * loss_recon
                     )
        
        return total_loss
            
            
    
    
    
    
    