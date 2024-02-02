"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.graphsage_px_net import GraphSagePxNet
from nets.mlp_net import MLPNet
from nets.pip_net import PIPNet
from nets.protgnn_net import ProtGNNNet
from nets.protop_gnn_net import ProtoPGNNNet
from nets.protop_gnn_global_net import ProtoPGNNgNet
from nets.tes_gnn_net import TesGNNNet
from nets.tes_gnn_global_net import TesGNNgNet
from nets.pxgnn_net import PxGNNNet
from nets.pip_gc_net import PIPgcNet
from nets.pip_gs_net import PIPgsNet
from nets.pp_net import PPNet


def GraphSage(net_params):
    return GraphSageNet(net_params)

def GraphSagePx(net_params):
    return GraphSagePxNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def PIP(net_params):
    return PIPNet(net_params)

def PP(net_params):
    return PPNet(net_params)

def PIPgc(net_params):
    return PIPgcNet(net_params)

def PIPgs(net_params):
    return PIPgsNet(net_params)

def ProtGNN(net_params):
    return ProtGNNNet(net_params)

def ProtoPGNN(net_params):
    return ProtoPGNNNet(net_params)

def ProtoPGNNg(net_params):
    return ProtoPGNNgNet(net_params)

def TesGNN(net_params):
    return TesGNNNet(net_params)

def TesGNNg(net_params):
    return TesGNNgNet(net_params)

def PxGNN(net_params):
    return PxGNNNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphSage': GraphSage,
        'GraphSagePx': GraphSagePx,
        'GCN': GCN,
        'GAT': GAT,
        'PIP': PIP,
        'PP': PP,
        'PIPgc': PIPgc,
        'PIPgs': PIPgs,
        'ProtGNN': ProtGNN,
        'ProtoPGNN': ProtoPGNN,
        'ProtoPGNNg': ProtoPGNNg,
        'TesGNN': TesGNN,
        'TesGNNg': TesGNNg,
        'PxGNN': PxGNN,
        'MLP': MLP
    }
        
    return models[MODEL_NAME](net_params)