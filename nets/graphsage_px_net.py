import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSagePxNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']    
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.device = net_params['device']
        self.readout = net_params['readout']
        self.lambda_recon = net_params['lambda_recon']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual) for _ in range(n_layers-1)])
        
        # prototype generator
        self.dec1 = MLPReadout(hidden_dim,in_dim)
        self.dec2 = nn.Linear(2*in_dim,1)
        
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
    def prot_generate(self, emb):
        x = self.dec1(emb) # N,D
        x_l = x.unsqueeze(1).repeat(1,x.size(0),1) # N,N,D
        x_r = x.unsqueeze(0).repeat(x.size(0),1,1) # N,N,D
        x_cat = self.dec2(torch.cat((x_l,x_r),dim=2)) # N,N,2*D
        
        s =  torch.sigmoid(x_cat.squeeze()) # N,N
        
        return s, x
    
    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        # reconstruction
        self.recon_hs = []
        self.recon_ss = []
        self.ind_graphs = dgl.unbatch(g)
        for one_g in self.ind_graphs:
            s, x = self.prot_generate(one_g.ndata['h'])
            self.recon_hs.append(x)
            self.recon_ss.append(s)
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return torch.sigmoid(self.MLP_layer(hg).squeeze())

        
    def loss(self, pred, label):
        criterion = nn.BCELoss()
        loss = criterion(pred, label)
        
        nb = label.size(0) # number of batch
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
        
        return loss + (loss_recon*self.lambda_recon)
    