"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import copy
import dgl
import numpy as np
import logging
logging.captureWarnings(True)
from sklearn.cluster import KMeans

from metrics import binary_f1_score, auc_score

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_auc, epoch_train_f1 = 0, 0
    nb_data = 0
    train_gs = []
    train_ys = []
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device).float()  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device).float()
        batch_graphs = batch_graphs.to(device)
        batch_targets = batch_targets.to(device).float() # batch
        optimizer.zero_grad()
    
        outputs = model.forward(batch_graphs, batch_x, batch_e)
        batch_scores = outputs
        loss = model.loss(batch_scores, batch_targets)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.detach().item()
        epoch_train_auc += auc_score(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_auc /= (iter + 1)
    #epoch_train_f1 /= (iter + 1)

    return epoch_loss, epoch_train_auc, optimizer
    
def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_auc, epoch_test_f1 = 0, 0
    epoch_scores, epoch_targets = [], []
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device).float()
            batch_e = batch_graphs.edata['feat'].to(device).float()
            batch_graphs = batch_graphs.to(device)
            batch_targets = batch_targets.to(device).float()
            
            outputs = model.forward(batch_graphs, batch_x, batch_e)
            batch_scores = outputs
            if hasattr(model,'attn_loss'):
                loss = model.loss(batch_scores, batch_targets) + model.attn_loss(batch_graphs)
            else:
                loss = model.loss(outputs, batch_targets)
       
            
            epoch_test_loss += loss.detach().item()
            epoch_test_auc += auc_score(batch_scores, batch_targets)
            #epoch_test_f1 += binary_f1_score(batch_scores, batch_targets)

            nb_data += batch_targets.size(0)
            epoch_scores.append(batch_scores)
            epoch_targets.append(batch_targets)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_auc /= (iter + 1)
        #epoch_test_f1 /= (iter + 1)

        
    #return epoch_test_loss, epoch_test_auc, epoch_test_f1
    return epoch_test_loss, epoch_test_auc

def _get_embeddings_for_init(model_fc, model_gnn, device, dataset, net_params):
    model_gnn.eval()
    model_fc.eval()
    model_fc.to(device)
    model_gnn.to(device)
    
    hgs_chd, hgs_con = [], []
    ys_chd, ys_con = [], []
    hs = []
    
    with torch.no_grad():
        for idx, (g,l) in enumerate(dataset):

            h = g.ndata['feat'].to(device).float()
            g = g.to(device)
            y = torch.tensor(np.array(l),dtype=torch.float).to(device).float()

            h = model_fc(h)
            for conv in model_gnn:
                h = conv(g,h)

            g.ndata['h'] = h    
            hg = dgl.mean_nodes(g, 'h')
            hs.append(h)

            if y == 0:
                hgs_chd.append(hg)
            elif y == 1:
                hgs_con.append(hg)

        hgs_chd_np = torch.vstack(hgs_chd).detach().cpu().numpy()   
        hgs_con_np = torch.vstack(hgs_con).detach().cpu().numpy()   

        kmeans_chd = KMeans(
                        init="random",
                        n_clusters=net_params['num_prot_per_class'],
                        n_init=10,
                        max_iter=1000,
                        random_state=41
                    )
        kmeans_con = KMeans(
                        init="random",
                        n_clusters=net_params['num_prot_per_class'],
                        n_init=10,
                        max_iter=1000,
                        random_state=41
                    )

        kmeans_chd.fit(hgs_chd_np)
        kmeans_con.fit(hgs_con_np)
        
        centroid_chd = kmeans_chd.cluster_centers_
        centroid_con = kmeans_con.cluster_centers_  

        pos = torch.tensor(centroid_con,dtype=torch.float32).to(device) # K,D           
        neg = torch.tensor(centroid_chd,dtype=torch.float32).to(device) # K,D
    
        # find nearest samples
        g_inits_pos, g_inits_neg = [], []
        h_inits_pos, h_inits_neg = [], []
        for prot in range(net_params['num_prot_per_class']):
            d_pos, d_neg = 10000, 10000
            for idx, (g,l) in enumerate(dataset):
                y = torch.tensor(np.array(l),dtype=torch.float).to(device).float()
                g = g.to(device)
                hg = torch.mean(hs[idx],dim=0)
                l2_pos = torch.norm(hg-pos[prot,:],dim=0,p=2)
                if l2_pos < d_pos and y == 1:
                    d_pos = l2_pos
                    g_pos = g
                    h_pos = hs[idx].clone()
            
                l2_neg = torch.norm(hg-neg[prot,:],dim=0,p=2)
                if l2_neg < d_neg and y == 0:
                    d_neg = l2_neg
                    g_neg = g
                    h_neg = hs[idx].clone()
                
            g_inits_pos.append(g_pos)
            g_inits_neg.append(g_neg)

            h_pos.requires_grad_(True)
            h_neg.requires_grad_(True)
            h_inits_pos.append(h_pos)
            h_inits_neg.append(h_neg)

    return g_inits_pos, g_inits_neg, h_inits_pos, h_inits_neg

def train_epoch_sparse_multi(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_auc, epoch_train_f1 = 0, 0
    nb_data = 0
    gpu_mem = 0
    for iter, (multi_batch_graphs, batch_targets) in enumerate(data_loader):
        batch_xs = [ batch_graphs.ndata['feat'].to(device).float() for batch_graphs in multi_batch_graphs ]  # (num x feat) x 8 list
        multi_batch_graphs = [ batch_graphs.to(device) for batch_graphs in multi_batch_graphs ]
        batch_targets = batch_targets.to(device).float()
        optimizer.zero_grad()
    
        outputs = model.forward(multi_batch_graphs, batch_xs)
        batch_scores = outputs
        loss = model.loss(outputs, batch_targets)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_auc += auc_score(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_auc /= (iter + 1)

    return epoch_loss, epoch_train_auc, optimizer

def evaluate_network_sparse_multi(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_auc, epoch_test_f1 = 0, 0
    epoch_scores, epoch_targets = [], []
    nb_data = 0
    with torch.no_grad():
        for iter, (multi_batch_graphs, batch_targets) in enumerate(data_loader):
            batch_xs = [ batch_graphs.ndata['feat'].to(device).float() for batch_graphs in multi_batch_graphs ]  # (num x feat) x 8 list
            multi_batch_graphs = [ batch_graphs.to(device) for batch_graphs in multi_batch_graphs ]
            batch_targets = batch_targets.to(device).float()
            
            outputs = model.forward(multi_batch_graphs, batch_xs)
            batch_scores = outputs
            loss = model.loss(outputs, batch_targets)
            
            epoch_test_loss += loss.detach().item()
            epoch_test_auc += auc_score(batch_scores, batch_targets)

            nb_data += batch_targets.size(0)
            epoch_scores.append(batch_scores)
            epoch_targets.append(batch_targets)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_auc /= (iter + 1)

    return epoch_test_loss, epoch_test_auc

