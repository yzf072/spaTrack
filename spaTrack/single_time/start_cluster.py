import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import seaborn as sns

import warnings
from IPython.display import display



def assess_start_cluster(adata):
    """Assess the entropy value to identify the starting cluster

    Parameters
    ----------
    adata
        Anndata

    Returns
    -------
        adata
    """
    


    cluster_name_list=list()
    entropy_list=list()
    cell_name_list=list()
    for i in range(0,len(adata.obs.index)):
        cell_id=adata.obs.index[i]
        cluster_name=adata.obs['cluster'][i]
        adata_cluster=adata[adata.obs.index.isin([cell_id])]
        matrix = np.array(adata_cluster.X)
        entropy_name=entropy(matrix[0])
        cluster_name_list.append(cluster_name)
        cell_name_list.append(cell_id)
        entropy_list.append(entropy_name)
    df_value=pd.DataFrame({'cluster_name': cluster_name_list,'cell_name':cell_name_list,'entropy':entropy_list})
    adata.obs['entropy']=df_value['entropy'].values

    df_obs=adata.obs
    cluster_order=list(pd.DataFrame(df_obs[['cluster','entropy']].groupby(['cluster']).mean()).sort_values(['entropy'],ascending=False).index)
    df_entropy=pd.DataFrame(df_obs[['cluster','entropy']].groupby(['cluster']).mean()).sort_values(['entropy'],ascending=False)

    df_obs['cluster']= pd.Categorical(df_obs['cluster'], categories=cluster_order, ordered=True)
    print('Cluster order sorted by entropy value: ',list(df_entropy.index))
    df_entropy.index=list(df_entropy.index)
    adata.uns['entropy value']=df_obs
    adata.uns['entropy value order']=df_entropy
    #display(df_entropy)
    return adata




def assess_start_cluster_plot(adata):
    """ Plot the entropy value and Stem Cell Differentiation score of each cluster

    Parameters
    ----------
    adata
        Anndata

    Returns
    -------
        figure
    """
    df_color=pd.DataFrame({"cluster":list(adata.obs['cluster'].cat.categories),"color_name":adata.uns['cluster_colors']})
    df_color.index=list(df_color['cluster'])
    color_palette=dict(zip(df_color['cluster'], df_color['color_name']))
    plt.figure(figsize=(10,9))
    ax1=sns.boxplot(data=adata.uns['entropy value'],x='cluster',y='entropy',linewidth=0.8,
                    palette=color_palette,
                   order=adata.uns['entropy value order'].index,showfliers=False)

    ax1.set_xticklabels(adata.uns['entropy value order'].index.values,rotation=40,ha='center',fontsize=15)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('entropy value',fontsize=20)


    plt.tight_layout()




    
    
        
    
    
        
