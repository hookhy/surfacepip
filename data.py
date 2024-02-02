import torch
import pickle
import time
import subprocess
import sys, os, glob
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy import sparse

import dgl

from sklearn.model_selection import StratifiedKFold, train_test_split

# load and make the dgl-format graph list
class LoadCHDdataset(torch.utils.data.Dataset):
    def __init__(self, dataConfig):

        # read data
        demo = pd.read_table(dataConfig['demo_path'],sep=' ')
        if dataConfig['target'] == 'ventricle':
            demo_fil = demo[np.logical_or(demo['ventricle']=='BiV',demo['ventricle']=='SV') ].reset_index(drop=True)
            self.target = 'ventricle'
        elif dataConfig['target'] == 'norwood':
            demo_fil = demo[np.logical_or(demo['norwood']=='0',demo['norwood']=='1') ].reset_index(drop=True)
            self.target = 'norwood'
        else:
            demo_fil = demo
            self.target = 'all'
        
        # including HCP 80
        if dataConfig['use_hcp'] == True:
            hcp_demo = pd.read_table(dataConfig['hcp_demo_path'],sep=' ')
            demo_fil = pd.concat([demo_fil,hcp_demo],axis=0).reset_index(drop=True)
        
        # filtering an incomplete data : 02-416-5
        #demo_fil = demo[demo["id"].str.contains("02-416-5")==False]
        
        # subject list
        subj_list = demo_fil['id'].to_numpy()
        n_subjects = len(subj_list)

        self.graph_lists = []
        self.graph_labels = []
        self.graph_sub_labels = []
        for i in range(n_subjects):
            s = str(subj_list[i])
            adjs_name = dataConfig['net_path'] + '/' + s + '_w.cgraph_adj.txt'
            nfeats_name = dataConfig['net_path'] + '/' + s + '_w.cgraph_nfeat.txt'

            if os.path.isfile(adjs_name):
                #print(s)
                adj = pd.read_table(adjs_name,header=None,sep=' ').to_numpy()
                nfeat = pd.read_table(nfeats_name,header=None,sep=' ').to_numpy()
                self.graph_lists.append(to_dgl_graph(adj, nfeat))
                
                if dataConfig['target'] == 'CHD':
                    self.graph_labels.append(demo_fil['group'][i]=='Control')
                elif dataConfig['target'] == 'ventricle':
                    self.graph_labels.append(demo_fil['ventricle'][i]=='BiV')
                elif dataConfig['target'] == 'norwood':
                    self.graph_labels.append(demo_fil['norwood'][i]=='1')
                self.graph_sub_labels.append(demo_fil['subgroup'][i])
                
            else:
                continue
        self.graph_sub_labels = (np.unique(self.graph_sub_labels, return_inverse=True)[1]).astype('int')
        
    def __getitem__(self, idx):
        """Get the idx-th sample.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)
    
def to_dgl_graph(adj,nfeat):

    sa = sparse.coo_matrix(adj)
        
    dgl_g = dgl.graph((np.array(sa.row), np.array(sa.col)),num_nodes=np.shape(adj)[0])
    dgl_bg = dgl.to_bidirected(dgl_g)
    
    m, s = np.mean(nfeat,axis=0), np.std(nfeat,axis=0)
    z_nfeat = (nfeat - m) / s
    dgl_bg.ndata['attr'] = torch.from_numpy(nfeat)
    dgl_bg.ndata['feat'] = torch.from_numpy(z_nfeat)
        
    return dgl_bg
    
# construct the train/val/test dataset using the dgl-format graph list
class CHDDataset(torch.utils.data.Dataset):
    def __init__(self, dataConfig):
        dataset = LoadCHDdataset(dataConfig)
        if dataConfig['subgroup'] == True:
            self.all_idx = get_all_split_idx_sub(dataset)
        else:
            self.all_idx = get_all_split_idx(dataset)
        
        self.all = dataset
        self.train = [format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(10)]
        self.val = [format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(10)]
        self.test = [format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(10)]
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels),dtype=torch.float)
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    
    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)    
    
def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
    """
    root_idx_dir = '/nasdata2/khj/gnns/project_chd/' + dataset.target + '/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    
    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + 'train.index')):
        print("[!] Splitting the data into train/val/test ...")
        
        # Using 10-fold cross val to compare with benchmark papers
        k_splits = 10

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []
        
        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
            
        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
            remain_index, test_index = indexes[0], indexes[1]    

            remain_set = format_dataset([dataset[index] for index in remain_index])
            
            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                    range(len(remain_set.graph_lists)),
                                                    test_size=0.111,
                                                    stratify=remain_set.graph_labels)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + 'train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + 'val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + 'test.index', 'a+'))
            
            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")
        
    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx    
    
def format_dataset(dataset): 

    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim).float()

    return DGLFormDataset(graphs, labels)

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    #new_g = dgl.DGLGraph() # deprecated
    new_g = dgl.graph([])
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

class MultiLobeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_ds = len(self.datasets)
    
    def __getitem__(self, idx):
        return [ tuple(d[idx]) for d in self.datasets ]
        
    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    def multi_collate(self, samples_list):
        multi_lobe_graphs = []
        for i in range(self.num_ds):
            one_lobe_list = []
            for sample in samples_list:
                one_lobe_list.append(sample[i])
            graphs, labels = map(list, zip(*one_lobe_list))
            labels = torch.tensor(np.array(labels),dtype=torch.float)
            batched_graph = dgl.batch(graphs)
            multi_lobe_graphs.append(batched_graph)
        
        return multi_lobe_graphs, labels

def get_all_split_idx_sub(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
    """
    root_idx_dir = '/nasdata2/khj/gnns/project_chd/sub/'#
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    
    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + 'train.index')):
        print("[!] Splitting the data into train/val/test ...")
        
        # Using 10-fold cross val to compare with benchmark papers
        k_splits = 10

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []
        
        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
            
        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_sub_labels):#
            remain_index, test_index = indexes[0], indexes[1]    

            remain_set = format_dataset([dataset[index] for index in remain_index])
            remain_graph_sub_labels = [ dataset.graph_sub_labels[index] for index in remain_index]#
            
            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                    range(len(remain_set.graph_lists)),
                                                    test_size=0.111,
                                                    stratify=remain_graph_sub_labels)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + 'train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + 'val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + 'test.index', 'a+'))
            
            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")
        
    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx    







    