import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

import ot
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state
from ot.backend import get_backend
from ot.gromov import init_matrix, gwloss, gwggrad

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def MAPE(scores, targets):
    scores = scores.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return np.mean(np.abs((targets - scores) / targets))

def PearsonRP(scores, targets):
    scores = scores.detach().cpu().numpy().astype(np.float32).reshape(-1,)
    targets = targets.detach().cpu().numpy().astype(np.float32).reshape(-1,)
    return pearsonr(scores, targets)

def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().detach().numpy()
    y_pred = scores.cpu().detach().numpy()
    y_pred = (y_pred >= 0.5) * 1
    return f1_score(y_true, y_pred)

def auc_score(scores, targets):
    """Computes the AUC score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().detach().numpy()
    y_pred = scores.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)

def parallel_fused_gromov_wasserstein2_learnablealpha( C1, C2, F1, F2, M, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, G0=None, **kwargs):
    r"""
    Computes the FGW distance between two graphs see (see :ref:`[24] <references-fused-gromov-wasserstein2>`)
    .. math::
        \min_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F + \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}
             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}
             \mathbf{\gamma} &\geq 0
    where :
    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as
    discussed in :ref:`[24] <references-fused-gromov-wasserstein2>`
    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.
    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p :  array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    log : bool, optional
        Record log if True.
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.
    Returns
    -------
    fgw-distance : float
        Fused gromov wasserstein distance for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.
    .. _references-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, F10, F20, M0, alpha0 = p, q, C1, C2, F1, F2, M, alpha
    nx = get_backend(p0, q0, C10, C20, F10, F20, M0, alpha0)


    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    F1 = nx.to_numpy(F10)
    F2 = nx.to_numpy(F20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)
    
    return fgw_dist




