import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import dgl

"""
Unofficial implementaion of ProtGNN
    Zhang, Zaixi et al., ProtGNN: Towards Self-Explaining Graph Neural Networks (AAAI 2022)
    https://arxiv.org/abs/2112.00911
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class ProtGNNNet(nn.Module):
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
        self.lambda_clst = net_params['lambda_clst']
        self.lambda_div = net_params['lambda_div']
        self.lambda_sep = net_params['lambda_sep']
        self.incorrect_strength = net_params['incorrect_strength']
        self.s_max = 0.3
        
        # prototype initilazation
        # (K,dim) for each class
        num_prototypes = int(self.num_prot_per_class*2)
        self.p_pos = torch.nn.Parameter( 
            torch.rand(
                (self.num_prot_per_class,hidden_dim), dtype=torch.float32, device=self.device
            )
            ,requires_grad=True 
        )
        self.p_neg = torch.nn.Parameter(
            torch.rand(
                (self.num_prot_per_class,hidden_dim), dtype=torch.float32, device=self.device
            )
            ,requires_grad=True 
        )

        
        # initial embedding layers
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # GNN embedding layers
       
        self.encoders = nn.ModuleList([
            GraphSageLayer(hidden_dim, hidden_dim, F.relu, dropout, 
                           aggregator_type, self.batch_norm, self.residual) for _ in range(n_layers)]) 

        # FC layers
        self.FC_layers = nn.Linear(num_prototypes,n_classes,bias=False)
        if self.incorrect_strength is not None:
            self.prototype_class_identity = torch.zeros(num_prototypes,n_classes)
            for j in range(num_prototypes):
                self.prototype_class_identity[j % self.num_prot_per_class,0] = 1
            self.set_last_layer_incorrect_connection(incorrect_strength=self.incorrect_strength)
    
    # set weights in last layer for better prototype learning
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.FC_layers.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
    
    # get similarity value: hgi - size of (batch,dim), prototype - size of (dim)
    def get_sims_prototypes(self, hgi, prototype):
        distance = torch.norm(hgi - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + 1e-12))
        
        return similarity # size of (batch)
        
    def forward(self, g, h, e):
            
        # feature initial embedding: input_dim -> hidden_dim
        h = self.embedding_h(h)
     
        # GNN embedding: hidden_dim -> hidden_dim
        for conv in self.encoders:
            h = conv(g, h)   
        g.ndata['h'] =  h   
        self.gs = g
        self.hg = dgl.mean_nodes(g, 'h') 

        # prototype projection    
        S_pos, S_neg = [], []
        for prot in range(self.num_prot_per_class):
            S_pos.append(self.get_sims_prototypes(self.hg, self.p_pos[prot,:]))
            S_neg.append(self.get_sims_prototypes(self.hg, self.p_neg[prot,:]))
        
        ss = torch.cat((torch.hstack(S_pos), torch.hstack(S_neg)),dim=1) # (batch,2*K)
        y = self.FC_layers(ss) # (batch,1)

        return torch.sigmoid(y).squeeze() # (batch)
    
    def loss(self, pred, label):
        
        nb = label.size(0) # number of batch
        # supervised loss
        criterion = nn.BCELoss()
        sup_loss = criterion(pred, label)

        # cluster loss
        norm_clst, norm_sep = 0, 0
        # for inverse projection,
        for i in range(nb):
            h_one = self.hg[i,:].unsqueeze(0).repeat(self.num_prot_per_class,1) # (num_prots,dim)
                
            if label[i] == 1:
                norm_clst +=  torch.min(torch.norm(h_one-self.p_pos,p=2,dim=1))
                norm_sep +=  torch.min(torch.norm(h_one-self.p_neg,p=2,dim=1))
                    
            elif label[i] == 0:
                norm_clst +=  torch.min(torch.norm(h_one-self.p_neg,p=2,dim=1))
                norm_sep +=  torch.min(torch.norm(h_one-self.p_pos,p=2,dim=1))

        norm_clst /= nb
        norm_sep /= nb
        
        # diversity loss
        loss_div = 0
        norm_p_pos = F.normalize(self.p_pos,dim=1,p=2)
        norm_p_neg = F.normalize(self.p_neg,dim=1,p=2)

        cossim_pos = torch.mm(norm_p_pos,norm_p_pos.t()) - self.s_max
        cossim_neg = torch.mm(norm_p_neg,norm_p_neg.t()) - self.s_max

        cossim_pos *= (cossim_pos>0).float()
        cossim_neg *= (cossim_neg>0).float()

        loss_div += torch.sum(torch.triu(cossim_pos, diagonal=1))
        loss_div += torch.sum(torch.triu(cossim_neg, diagonal=1))

        total_loss = ( 
                        sup_loss 
                      + self.lambda_clst * norm_clst 
                      - self.lambda_sep * norm_sep
                      + self.lambda_div * loss_div
       
                     )
        
        return total_loss
            
            
    
    
    
    
    