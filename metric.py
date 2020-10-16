mport numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt


class Metric:
    def __init__(self, dim):
        self.dim = dim 
        self.metric = None
    """
    attributes:
        self.dim: int; dimension
        self.metric: dim x dim np.array;
    """
    
    def get_metric(self, dim=None, s_dim=2, s_var=1e-3, l_var=0.5, l_mean=0.2, should_round=None):
        """
        constructs a Fubini-Study metric with some converging eigenstates,
        for which entries vanish in rows and columns and variance is small
        In:
            dim: int; matrix dimension only required if different to self.dim
            s_dim: int; number of converging eigenstates
            s_var: float; variance of converging eigenstates
            l_var: float; variance of non-converging eigenstates
            l_mean: float; mean of non-converging eigenstates
            should_round: int; optional rounding of metric, None if unwanted 
            
        Out:
            dim x dim np.array: FB metric approximate
        """

        """get random matrix"""
        dim = dim or self.dim
        metric = np.random.normal(l_mean, np.sqrt(l_var), size=(dim, dim))
        
        """make symmetric"""
        metric = metric - np.triu(metric) + np.tril(metric).transpose()
        
        """add converging subsystem block"""
        singular_subsystem = np.random.normal(0., np.sqrt(s_var), size=(s_dim, s_dim))
        metric[dim-s_dim:, dim-s_dim:] = singular_subsystem
        
        """adjust non-converging diagonal entries"""
        non_converging_diagonal = 1. - np.random.normal(l_mean, np.sqrt(l_var), size=(dim-s_dim))
        np.fill_diagonal(metric[:dim-s_dim, :dim-s_dim], non_converging_diagonal)
        
        """add interaction matrix between convergent and non-convergent parts"""
        interaction_matrix = np.random.normal(0., np.sqrt(l_var), size=(s_dim,dim-s_dim))
        metric[dim-s_dim:, :dim-s_dim] = interaction_matrix
        metric[:dim-s_dim, dim-s_dim:] = interaction_matrix.transpose()
        
        """rounding"""
        if should_round is not None:
            metric = np.around(metric, decimals=should_round)
        
        self.metric = metric
    

    def block_truncate(self, b_size=3):
        """
        truncation of metric into blocks of size b_size
        
        In:
            b_size: int; block sizes
        
        Out:
            dim x dim np.array: truncated matrix
        """
        
        metric = self.metric
        dim = len(metric)
        
        for i in range(int(dim/b_size)):
            patch = np.zeros((dim-(i+1)*b_size, b_size))
            
            metric[b_size*(i+1):, b_size*i:b_size*(i+1)] = patch
            metric[b_size*i:b_size*(i+1), b_size*(i+1):] = patch.transpose()
            
        self.metric = metric
    
    
    def regularize(self, reg_param=0.05):
        """
        Tikhonov regularization
        
        In:
            reg_param: float; regularization parameter
            
        Out:
            dim x dim np.array: padded matrix
        """
        
        self.metric = self.metric + reg_param * np.identity(len(self.metric))

    
    def plot_spectrum(self, file=None, title=''):
        """
        Plots eigenvalue spectrum of self.metric in a sleek way
        In:
            file: str; if desired save filename o/w None
            
        Out:
            plt.figure
            png with path file
        """
        
        plt.figure()
        fig, ax = plt.subplots(1,1, figsize=(10, 5), sharex=True)
        
        spectrum = np.linalg.eigvals(self.metric)

        for eigval in spectrum:
            ax.vlines(eigval, 0., 1., colors='royalblue', linestyles='solid', linewidth=2.)

        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('eigenvalues')
        ax.grid(True)
        ax.set_ylim(0., 2.)
        ax.set_title(title+' eigenvalues')
        
        if file is not None:
            plt.savefig(file, dpi=400)      
        
        plt.show()


def get_overlap(self, metric=None, cutoff=0):
        """
        computes the inner products of eigenstate-pairs of a) intra-group of 
        non-singular parts of the metric b) intra-group of singular parts and c) 
        inter-group between both previous groups
        
        In:
            metric: dim x dim np.array; if not working on self.metric
            cutoff: float: condition abs(eigval) < cutoff separated into the groups
            
        Out:
            list for pairs a)
            list for pairs b)
            list for pairs c)
        """        
        
        F = metric or self.metric
        eigvals, eigvecs = np.linalg.eig(F)
        
        """separate into groups"""
        sing_vecs = eigvecs[eigvals <= cutoff]
        nonsing_vecs = eigvecs[eigvals > cutoff]
        
        """get scalar products"""
        inners_sing = []
        for i, vecs in enumerate(sing_vecs[:-1]):
            inners_sing += [np.inner(vecs, other) for other in sing_vecs[i+1:]]
            
        inners_non = []
        for i, vecs in enumerate(nonsing_vecs[:-1]):
            inners_non += [np.inner(vecs, other) for other in nonsing_vecs[i+1:]]
            
        inners_inter = []
        for vecs in nonsing_vecs:
            inners_inter += [np.inner(vecs, other) for other in sing_vecs]

        return np.array(inners_sing), np.array(inners_non), np.array(inners_inter)

