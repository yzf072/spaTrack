import numpy as np
import anndata as ad
from anndata import AnnData
from scipy.sparse import issparse
from typing import List, Tuple, Optional
import ot
import pandas as pd

from .utils import gene_dist,spatial_dist,get_exp_matrix,pre_check_adata


def transfer_matrix(
    adata1: AnnData, 
    adata2: AnnData, 
    layer: str = "X",
    spatial_key: str = "spatial",
    alpha: float = 0.1, 
    spa_method: str = 'euclidean',
    gene_method: str ='kl', 
    distribution_1 = None,
    distribution_2 = None, 
    G_0 = None,
    numItermax: int = 200, 
    numItermaxEmd: int = 1000000,
    verbose: bool = True,  
    **kwargs
):
    
    
    """
    Calculates transfer matrix between two time. 
    
    Args:
        adata1: the first time(source) data.
        adata2: the second time(target) data.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        spatial_key: Key in .obsm containing coordinates for each cell.
        alpha:  Alignment tuning parameter. Note:0 <= alpha <= 1. When ``alpha = 0`` only the gene expression data is taken into account,
               while ``alpha =1`` only the spatial coordinates are taken into account.
        spa_method: calculate spatial coordinate distances, defult is euclidean.
        gene_method: calculate gene expression dissimilarity measure: ``'euclidean'`` or ``'cosine'``or``'wasserstein'``or``'kl'``.
        distribution_1: The probability distribution of cells(in adata1),defult is None,means uniform distribution.
        distribution_2: The probability distribution of cells(in adata2),defult is None,means uniform distribution.
        G_0: The joint density distribution,defult is independent.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
   
    Returns:
        matrix(dim:n_obs*m_obs) of transition probability.
    """
    ## data process
    common_genes = list(set(adata1.var.index)&set(adata2.var.index))
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    
    ## get gene expression matrix
    exp_martix_1 = get_exp_matrix(adata1)
    exp_martix_2 = get_exp_matrix(adata2)
    
    ## Calculate spatial coordinates diatances
    spa_dist_1 = spatial_dist(adata1,spa_method='euclidean')
    spa_dist_2 = spatial_dist(adata2,spa_method='euclidean')
    
    ## Calculate gene expression dissimilarity
    M = gene_dist(exp_martix_1,exp_martix_2,gene_method='kl')
    
    ## init distributions
    p_1 = np.ones((adata1.shape[0],)) / adata1.shape[0] if distribution_1 is None else np.asarray(distribution_1)
    p_2 = np.ones((adata2.shape[0],)) / adata2.shape[0] if distribution_2 is None else np.asarray(distribution_2)
    
    ## Run FGW-OT
    constC, hC1, hC2 = ot.gromov.init_matrix(spa_dist_1, spa_dist_2, p_1, p_2, loss_fun="square_loss")
    ## defult independent joint density.
    if G_0 is None:
        G0 = p_1[:,None] * p_2[None, :]
    ## gwloss+gwgrad
    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)
    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)
    
    pi,log = ot.gromov.cg(
        p_1,
        p_2,
        (1 - alpha) * M, ## loss matrix
        alpha, ## Regularization term>0
        f,
        df,
        G0,
        armijo=False,
        C1=spa_dist_1,
        C2=spa_dist_2,
        constC=constC,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        log=True,)
    
    pi = np.array(pi)
    
    return pi


def generate_animate_input(
    pi_list:List=[np.array],
    adata_list:List=[AnnData],
    spatial_key:str='spatial',
    time:str='batch',
    annotation:str='Celltype'
):  
    """
    Generate animate transfer input of two or more times. 
    
    Args:
        pi_list: transfer matrix (nd.array) list.
        adata_list: AnnData list.
        spatial_key: Key in .obsm containing coordinates for each cell.
        time: time Key in .obs.
        annotation: cell type Key in .obs.
    Returns；
        DataFrame of transfer info between all times, as input of dispaly animate plot.

    """
    ## 1.check adata.obs contains keys:x,y,time.
    for i,adata in enumerate(adata_list):
        
        if adata.obs.columns.isin(['x','y','time','cell_id','annotation']).sum() == 5:
            pass
        else:
            adata_list[i]= pre_check_adata(adata,spatial_key=spatial_key,time=time,annotation=annotation)

    
    ## 2.merge transfer matrix data for two or more time, return pi_matrix, such as slice1,slice2,slice3,......
    map_list=[]
    for i in range(0,len(adata_list)-1):
        map_data,pi=__pi_process(pi_list[i],adata_list[i],adata_list[i+1])
        map_data['slice'+str(i+1)] =pi.index
        map_data['slice'+str(i+2)]=pi.idxmax(axis=1)
        map_data['pi_value'+str(i+1)]=pi.max(axis=1)
        map_list.append(map_data)
        if i >=1 and i <= len(adata_list)-1:
            map_list[i]=pd.merge(map_list[i-1], map_list[i], left_on='slice'+str(i+1), right_on='slice'+str(i+1)).drop(
                        ['pi_value'+str(i),'pi_value'+str(i+1)], axis=1)
    pi_matrix = map_list[-1]
    
    ## 3.merge adata.obs for two or more time.
    data_info = pd.concat(([i.obs for i in adata_list]),axis = 0)
    
    ## 4.merge pi_matrix & data_info, to add coord info and annotation
    for i, element in enumerate(map_list[-1].columns):
        pi_matrix = pd.merge(pi_matrix, data_info, left_on=element, right_on='cell_id').drop(['time','cell_id'], axis=1)
    
    ## 4.1 get sub columns,slice1,annotation,x,y
    pi_matrix_columns = pi_matrix.columns.to_list()   
    slice_list = map_list[-1].columns.to_list()
    annotation_list = [i for i in pi_matrix_columns if i.startswith("annotation")]
    x_list = [i for i in pi_matrix_columns if i.startswith("x")]
    y_list = [i for i in pi_matrix_columns if i.startswith("y")]
    lable_list = slice_list + annotation_list + x_list + y_list
    pi_matrix_coord = pi_matrix[lable_list]
    
    ## 4.2 rename sub solumns.
    annotation_dic = {ele:"slice"+str(i+1)+"_annotation" for i,ele in enumerate(annotation_list)}
    x_dic = {ele:"slice"+str(i+1)+"_x" for i,ele in enumerate(x_list)}
    y_dic = {ele:"slice"+str(i+1)+"_y" for i,ele in enumerate(y_list)}
    pi_matrix_coord=pi_matrix_coord.rename(columns={**annotation_dic,**x_dic,**y_dic})
   
    ## 4.3 calculate transfer line slop and intercept.
    for i, element in enumerate(slice_list):
        if i >=1 and i <= len(slice_list)-1:
            pi_matrix_coord['k_line_'+str(i)+str(i+1)]=(pi_matrix_coord["slice"+str(i)+"_y"] -pi_matrix_coord["slice"+str(i+1)+"_y"]) / (pi_matrix_coord["slice"+str(i)+"_x"] - pi_matrix_coord["slice"+str(i+1)+"_x"])
            pi_matrix_coord['b_line_'+str(i)+str(i+1)] = pi_matrix_coord["slice"+str(i)+"_y"] -pi_matrix_coord['k_line_'+str(i)+str(i+1)]*pi_matrix_coord["slice"+str(i)+"_x"]
    
    ## return pandas as animate_transfer input.
    return pi_matrix_coord

def __pi_process(
    pi,
    adata1,
    adata2,
):  
    """
    transfer matrix process to get DataFrame. 
    """
    pi = pd.DataFrame(pi)
    pi.index = adata1.obs.index
    pi.columns = adata2.obs.index
    map_data = pd.DataFrame(index=pi.index)
    return map_data,pi