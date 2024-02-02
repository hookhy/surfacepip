import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

import pickle
import time
import copy
import subprocess
import argparse, json
import re
import random
import sys, os, glob
import numpy as np
import pandas as pd
import csv
from scipy.io import loadmat
from scipy import sparse
import dgl

# custom modules
from nets.load_net import gnn_model # import GNNs
from data import CHDDataset as LoadData
from train_CHDclassification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
from train_CHDclassification import _get_embeddings_for_init
from layers.graphsage_layer import GraphSageLayer

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME,dataset, params, net_params, dirs):
    avg_test_auc, avg_train_auc = [],[]
    avg_convergence_epochs = []
    all_scores, all_targets = [], []

    t0 = time.time()
    per_epoch_time = []
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format('CHD', MODEL_NAME, params, net_params, net_params['total_param']))
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        
        for split_number in range(10):
            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            dgl.seed(params['seed'])
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
                torch.cuda.manual_seed_all(params['seed'])

            print("RUN NUMBER: ", split_number)
            
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            
            model = gnn_model(MODEL_NAME, net_params)
            if net_params['pretrained_path'] is not None and MODEL_NAME == 'PxGNN':
                
                # load the pre-trained gnn model
                # model consists of (embedding_fc, gnns, prediction_fc)
                final_path = os.path.join(net_params['pretrained_path'],'RUN_' + str(split_number))
                pkls = glob.glob(final_path + '/*')
                ckpt_pretrained = torch.load(pkls[-1],map_location=device) # load the weights
                
                # matching the weight keys 
                ckpt_embed = {key: val for key, val in ckpt_pretrained.items() if key.startswith('embedding_h')}
                ckpt_embed = dict((key.replace('embedding_h.',''), value) for (key, value) in ckpt_embed.items())
                ckpt_gnn = {key: val for key, val in ckpt_pretrained.items() if key.startswith('layers')}
                ckpt_gnn = dict((key.replace('layers.',''), value) for (key, value) in ckpt_gnn.items())
                ckpt_dec1 = {key: val for key, val in ckpt_pretrained.items() if key.startswith('dec1')}
                ckpt_dec1 = dict((key.replace('dec1.',''), value) for (key, value) in ckpt_dec1.items())
                ckpt_dec2 = {key: val for key, val in ckpt_pretrained.items() if key.startswith('dec2')}
                ckpt_dec2 = dict((key.replace('dec2.',''), value) for (key, value) in ckpt_dec2.items())
                
                model.embedding_h.load_state_dict(ckpt_embed)
                model.encoders.load_state_dict(ckpt_gnn)
                model.dec1.load_state_dict(ckpt_dec1)
                model.dec2.load_state_dict(ckpt_dec2)
                
                # initializing the prototypes
                model.g_pos_init,model.g_neg_init, model.h_pos_init, model.h_neg_init = _get_embeddings_for_init(model.embedding_h, model.encoders, device, trainset, net_params)
                
                #
                model.p_pos = torch.nn.ParameterList(
                    [ 
                        torch.nn.Parameter(model.h_pos_init[prot].clone(),requires_grad=True) for prot in range(model.num_prot_per_class) 
                    ]
                )
                model.p_neg = torch.nn.ParameterList(
                    [ 
                        torch.nn.Parameter(model.h_neg_init[prot].clone(),requires_grad=True) for prot in range(model.num_prot_per_class) 
                    ]
                )
            
            if net_params['pretrained_path'] is not None and net_params['pruning']:
                
                final_path = os.path.join(net_params['pretrained_path'],'RUN_' + str(split_number))
                pkls = glob.glob(final_path + '/*')
                ckpt_pretrained = torch.load(pkls[-1],map_location=device) # load the weights
                
                model.load_state_dict(ckpt_pretrained)
                for param in model.parameters():
                    param.requires_grad = False
                
                # merging last layer
                last_layer = model.FC_layers
                ws = last_layer.state_dict()['weight'] # (1,4)
                ws_hat = torch.zeros((1,3))
                ws_hat[0,0] = ws[0,0] + ws[0,3]# shared prot
                ws_hat[0,1] = ws[0,1]
                ws_hat[0,2] = ws[0,2]
                model.FC_layers = torch.nn.Linear(3,1,bias=False)
                model.FC_layers.weight = torch.nn.Parameter(ws_hat)
                
            model = model.to(device)
            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_aucs, epoch_val_aucs = [], []
     
            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=True, collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)
            
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)
            
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)    
                    start = time.time()
                    
                    epoch_train_loss, epoch_train_auc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    epoch_val_loss, epoch_val_auc = evaluate_network(model, device, val_loader, epoch)
                    _, epoch_test_auc = evaluate_network(model, device, test_loader, epoch)
                    
                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_aucs.append(epoch_train_auc)
                    epoch_val_aucs.append(epoch_val_auc)
                    
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_auc', epoch_train_auc, epoch)
                    writer.add_scalar('val/_auc', epoch_val_auc, epoch)
                    writer.add_scalar('test/_auc', epoch_test_auc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                   
                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_auc=epoch_train_auc, val_auc=epoch_val_auc,
                                  test_auc=epoch_test_auc)  
                     
                    per_epoch_time.append(time.time()-start)

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch-1:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break
                        
                    # Stop training after params['max_time'] hours
                    if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                        print('-' * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                        break
            
            _, test_auc = evaluate_network(model, device, test_loader, epoch)   
            _, train_auc = evaluate_network(model, device, train_loader, epoch)    
            avg_test_auc.append(test_auc)   
            avg_train_auc.append(train_auc)
            
            avg_convergence_epochs.append(epoch)

            print("Train AUC [LAST EPOCH]: {:.6f} , Test AUC [LAST EPOCH]: {:.6f}".format(train_auc,test_auc))
            print("Convergence Time (Epochs): {:.6f}".format(epoch))
            
            if params['run_to'] is not None and split_number >= params['run_to']:
                print("\n!! K-FOLD EQUAL TO MAXIMUM FOLD SET.")
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
          
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(avg_convergence_epochs))))
    print("""\n\n\nFINAL RESULTS\n\nTEST AUC averaged: {:.6f} with s.d. {:.6f}""".format(np.mean(np.array(avg_test_auc)), np.std(avg_test_auc)))
    print("\nAll splits Test AUCs:\n", avg_test_auc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN AUC averaged: {:.6f} with s.d. {:.6f}""".format(np.mean(np.array(avg_train_auc)), np.std(avg_train_auc)))
    print("\nAll splits Train AUCs:\n", avg_train_auc)
    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
       
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST AUC averaged: {:.6f} with s.d. {:.6f}\nTRAIN AUC averaged: {:.6f} with s.d. {:.6f}\n\n
    Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test AUCs: {}"""\
          .format('HYU-PRETERM', MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_auc)), np.std(avg_test_auc),
                  np.mean(np.array(avg_train_auc)), np.std(avg_train_auc),
                  np.mean(avg_convergence_epochs), np.std(avg_convergence_epochs),
               (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_auc))         
        
def main():    

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--target', help="Please give a value for target")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--net_path', help="Please give a value for net_path")
    parser.add_argument('--demo_path', help="Please give a value for demopath")
    parser.add_argument('--hcp_demo_path', help="Please give a value for demopath")
    parser.add_argument('--use_hcp', help="Please give a value for use_hcp")
    parser.add_argument('--pretrained_path', help="Please give a value for pretrained_path")
    parser.add_argument('--pruning', help="Please give a value for pruning")
    parser.add_argument('--subgroup', help="Please give a value for subgroup")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--run_to', help="Please give a value for run_to")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--ppr_k', help="Please give a value for ppr_k")
    parser.add_argument('--alpha', help="Please give a value for alpha")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--num_prot_per_class', help="Please give a value for num_prot_per_class")
    parser.add_argument('--lambda_clst', help="Please give a value for lambda_clst")
    parser.add_argument('--lambda_sep', help="Please give a value for lambda_sep")
    parser.add_argument('--lambda_div', help="Please give a value for lambda_div")
    parser.add_argument('--lambda_reg', help="Please give a value for lambda_reg")
    parser.add_argument('--lambda_clssep', help="Please give a value for lambda_clssep")
    parser.add_argument('--lambda_recon', help="Please give a value for lambda_recon")
    parser.add_argument('--incorrect_strength', help="Please give a value for incorrect_strength")
    parser.add_argument('--prot_decoding', help="Please give a value for prot_decoding")
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
        
    if args.net_path is not None:
        config['net_path'] = args.net_path
    if args.demo_path is not None:
        config['demo_path'] = args.demo_path
    if args.pretrained_path is not None:
        config['pretrained_path'] = args.pretrained_path
    if args.pruning is not None:
        config['pruning'] = args.pruning
    if args.subgroup is not None:
        config['subgroup'] = args.subgroup
    if args.use_hcp is not None:
        config['use_hcp'] = args.use_hcp
    if args.hcp_demo_path is not None:
        config['hcp_demo_path'] = args.hcp_demo_path
    if args.target is not None:
        config['target'] = args.target
    
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.run_to is not None:
        params['run_to'] = int(args.run_to)
    
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.ppr_k is not None:
        net_params['ppr_k'] = int(args.ppr_k)
    if args.alpha is not None:
        net_params['alpha'] = float(args.alpha)
    if args.num_prot_per_class is not None:
        net_params['num_prot_per_class'] = int(args.num_prot_per_class)
    if args.lambda_sep is not None:
        net_params['lambda_sep'] = float(args.lambda_sep)
    if args.lambda_clst is not None:
        net_params['lambda_clst'] = float(args.lambda_clst)
    if args.lambda_div is not None:
        net_params['lambda_div'] = float(args.lambda_div)
    if args.lambda_reg is not None:
        net_params['lambda_reg'] = float(args.lambda_reg)
    if args.lambda_clssep is not None:
        net_params['lambda_clssep'] = float(args.lambda_clssep)
    if args.lambda_recon is not None:
        net_params['lambda_recon'] = float(args.lambda_recon)
    if args.incorrect_strength is not None:
        net_params['incorrect_strength'] = float(args.incorrect_strength)
    if args.prot_decoding is not None:
        net_params['prot_decoding'] = bool(args.prot_decoding)   

    ##
    DATASET_CFG = dict()
    
    DATASET_CFG['net_path'] =  config['net_path']
    DATASET_CFG['demo_path'] =  config['demo_path']
    DATASET_CFG['subgroup'] =  config['subgroup']
    DATASET_CFG['use_hcp'] =  config['use_hcp']
    DATASET_CFG['hcp_demo_path'] =  config['hcp_demo_path']
    DATASET_CFG['target'] =  config['target']
    dataset = LoadData(DATASET_CFG)

    net_params['pretrained_path'] = config['pretrained_path']
    net_params['pruning'] = config['pruning']
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    net_params['n_classes'] = 1 # for bcewithlogit loss
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" +  "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" +  "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" +  "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" +  "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)
    
main()    








