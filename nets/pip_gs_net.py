import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import dgl

"""
PIP Net with global similarity.
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class PIPgsNet(nn.Module):
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
        self.lambda_reg = net_params['lambda_reg']
        self.incorrect_strength = net_params['incorrect_strength']
        
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

        # alignment layer for embedding co-representations between prots and hidden node features
        self.W1_layer = nn.Linear(2*hidden_dim,hidden_dim,bias=False)
        self.W2_layer = nn.Linear(hidden_dim,1,bias=False)
        
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
    
    # prototype inverse projection 
    def inverse_prot_proj(self, feats):
        prot_mat = torch.cat((self.p_pos,self.p_neg),dim=0) # (2*K,d)
    
        self.a = []
        out = []
        for feat in feats:
            # embed a co-representation for extracting inverse-projection matrix
            _feat = feat.unsqueeze(0).repeat(prot_mat.size(0),1,1) # (2*K,N,d), N: number of nodes for the subject
            _prot_mat = prot_mat.unsqueeze(1).repeat(1,feat.size(0),1) # (2*K,N,d)
            cat_mat = torch.cat((_feat,_prot_mat),dim=-1) # (2*K,N,2*d)
            att = self.W2_layer(torch.tanh(self.W1_layer(cat_mat))).squeeze() # (2*K,N)
            att = torch.sigmoid(att)
            self.a.append(att)
            
            # inverse projection of input features
            out.append(torch.mm(att,feat)) # (2*K,d)
        out = torch.stack(out,dim=1) # (2*K,batch,d)
        return out
        
    def forward(self, g, h, e):
            
        # feature initial embedding: input_dim -> hidden_dim
        h = self.embedding_h(h)
     
        # GNN embedding: hidden_dim -> hidden_dim
        for conv in self.encoders:
            h = conv(g, h)   
        g.ndata['h'] =  h   
        self.gs = g
        self.hg = dgl.mean_nodes(g, 'h') 
        
        # prototype layer
        # inverse prototype projection
        individual_graphs = dgl.unbatch(g)
        feat = [ one_g.ndata['h'] for one_g in individual_graphs ]
        self.feat_attn = self.inverse_prot_proj(feat) # (K,batch,d)
            
        # focal similarity
        S_pos, S_neg = [], []
        for prot in range(self.num_prot_per_class):
            S_pos.append(
                (
                    self.get_sims_prototypes(self.feat_attn[prot,:,:], self.p_pos[prot,:])
                )
            )
            S_neg.append(
                (
                    self.get_sims_prototypes(self.feat_attn[prot+self.num_prot_per_class,:,:], self.p_neg[prot,:])
                )
            )
        
        ss = torch.cat((torch.hstack(S_pos), torch.hstack(S_neg)),dim=1) # (batch,2*K)
        y = self.FC_layers(ss) # (batch,1)

        return torch.sigmoid(y).squeeze() # (batch)
    
    def loss(self, pred, label):
        
        nb = label.size(0) # number of batch
        # supervised loss
        criterion = nn.BCELoss()
        sup_loss = criterion(pred, label)

        # cluster loss
        norm_clst = 0
        h_pos, h_neg = [],[]
       
        for i in range(nb):
            attn_h = self.feat_attn[:,i,:] # (2*K,d)
            
            if label[i] == 1:
                attn_p = attn_h[:self.num_prot_per_class,:]
                norm_clst +=  torch.min(torch.norm(attn_p-self.p_pos,p=2,dim=1)) 
                h_pos.append(attn_p)
                    
            elif label[i] == 0:
                attn_n = attn_h[self.num_prot_per_class:,:]
                norm_clst +=  torch.min(torch.norm(attn_n-self.p_neg,p=2,dim=1)) 
                h_neg.append(attn_n)

        norm_clst /= nb
        
        # diversity loss
        loss_div = 0
        prot_mat = torch.cat((self.p_pos,self.p_neg),dim=0) # (num_prots,dim)
        if self.lambda_div != 0:
            I_k = torch.eye(prot_mat.size(0),device=self.device)
            loss_div += torch.norm(torch.mm(prot_mat,prot_mat.t())-I_k,p='fro')

        # subspace separation loss
        loss_reg = 0
        if self.lambda_reg != 0: 
            h_pos = torch.stack(h_pos,dim=0) # (n_pos,num_prots_per_class,dim)
            h_neg = torch.stack(h_neg,dim=0) # (n_neg,num_prots_per_class,dim)
            for prot in range(self.num_prot_per_class):
                _p_pos = self.p_pos[prot,:].unsqueeze(0).repeat(h_pos.size(0),1) # (n_pos,dim)
                _p_neg = self.p_neg[prot,:].unsqueeze(0).repeat(h_neg.size(0),1) # (n_neg,dim)
                loss_reg +=  torch.min(torch.norm(h_pos[:,prot,:]-_p_pos,p=2,dim=1))
                loss_reg +=  torch.min(torch.norm(h_neg[:,prot,:]-_p_neg,p=2,dim=1))
     
            loss_reg /= prot_mat.size(0)

        total_loss = ( 
                        sup_loss 
                      + self.lambda_clst * norm_clst 
                      + self.lambda_div * loss_div
                      + self.lambda_reg * loss_reg
       
                     )
        
        return total_loss
            
            
    
    
    
    
    