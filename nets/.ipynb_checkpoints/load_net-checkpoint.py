"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.mlp_net import MLPNet
from nets.appnp_net import APPNPNet
from nets.tot_net import TOTNet

def APPNP(net_params):
    return APPNPNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def TOT(net_params):
    return TOTNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'APPNP': APPNP,
        'GraphSage': GraphSage,
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'TOT': TOT,
        'MLP': MLP
    }
        
    return models[MODEL_NAME](net_params)