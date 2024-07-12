import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display



def assess_start_cluster(adata,mouse=False):
    """Assess the G2M score and Stem Cell Differentiation score to identify the starting cluster

    Parameters
    ----------
    adata
        Anndata
    mouse, bool
         Mouse is False represent using Capitalized gene symobol name

    Returns
    -------
        adata
    """
    ##part1: Compare G2M score of clusters 

    filepath=os.path.dirname(__file__)
    G2M_filepath=filepath.split('single_time')[0]+'example.data/regev_lab_cell_cycle_genes.txt'
    df_G2M_gene=pd.read_table(G2M_filepath)
    if mouse==False:
        pass
    else:
        df_G2M_gene['G2M_gene']=df_G2M_gene['G2M_gene'].str[0]+df_G2M_gene['G2M_gene'].str.lower().str[1:]

    s_genes = df_G2M_gene[:43]
    g2m_genes = df_G2M_gene[43:]
    df_G2M_gene = [x for x in df_G2M_gene['G2M_gene'] if x in adata.var_names]



    try:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes['G2M_gene'], g2m_genes=g2m_genes['G2M_gene'])

    except Exception as e:
        print(" Please use Gene Symbol in adata.var_names")
        raise e


    #sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    #sc.pp.scale(adata)

    df_obs=adata.obs
    cluster_order=list(pd.DataFrame(df_obs[['cluster','G2M_score']].groupby(['cluster']).mean()).sort_values(['G2M_score'],ascending=False).index)
    df_G2M_score=pd.DataFrame(df_obs[['cluster','G2M_score']].groupby(['cluster']).mean()).sort_values(['G2M_score'],ascending=False)

    df_obs['cluster']= pd.Categorical(df_obs['cluster'], categories=cluster_order, ordered=True)
    print('Cluster order sorted by G2M score: ',list(df_G2M_score.index))
    df_G2M_score.index=list(df_G2M_score.index)
    adata.uns['G2M score']=df_obs
    adata.uns['G2M score order']=df_G2M_score
    display(df_G2M_score)

    
    
    
    
    filepath=os.path.dirname(__file__)
    G2M_filepath=filepath.split('single_time')[0]+'example.data/GOBP_STEM_CELL_DIFFERENTIATION.v2023.2.Hs.tsv'
    df_pathway_gene=pd.read_table(G2M_filepath)

    #df=pd.read_table('/home/huangke2/huangke2_694/27.spatial.trajectory/12.SpaTrack/15.response/GOBP_STEM_CELL_DIFFERENTIATION.v2023.2.Hs.tsv')
    gene_names=df_pathway_gene.loc[df_pathway_gene['STANDARD_NAME']=='GENE_SYMBOLS']['GOBP_STEM_CELL_DIFFERENTIATION'].iloc[0].split(',')
    
    if mouse==False:
        merge_genes=set(pd.DataFrame(gene_names)[0].values).intersection(set(adata.var_names))
    else:
        df_pathway_gene=pd.DataFrame(gene_names)
        df_pathway_gene[0]=df_pathway_gene[0].str[0]+df_pathway_gene[0].str.lower().str[1:] 
        merge_genes=set(pd.DataFrame(df_pathway_gene[0])[0].values).intersection(set(adata.var_names))
    
    
    
    df_exp=pd.DataFrame(adata[:,  list(merge_genes)].X.toarray())
    df_exp.index=adata.obs['cluster']

    stem_score_list=list()
    for cluster_name in adata.obs['cluster'].value_counts().index:
        #print(cluster_name)
        stem_score=df_exp.loc[df_exp.index.isin([cluster_name])].mean(axis=1)
        stem_score_list.append(stem_score)
    df_res=pd.DataFrame(pd.concat(stem_score_list))
    df_res['cluster']=list(df_res.index.values)
    df_res.columns=['score','cluster']
    df_res.index=list(df_res['cluster'])

    ## output dataframe 
    cluster_order=df_res.groupby(['cluster']).mean().sort_values(by='score',ascending=False)
    cluster_order.index=list(cluster_order.index)
    cluster_order.columns=['Mean of Stem Cell Differentiation score']
    df_count=pd.DataFrame(adata.obs['cluster'].value_counts())
    df_count.columns=['cell_number']
    cluster_order['cell_number']=df_count.loc[cluster_order.index]['cell_number']
    #cluster_order['cluster']=cluster_order.index
    adata.uns['Stem Cell Differentiation score']=df_res
    adata.uns['Stem Cell Differentiation score order']=cluster_order
    print('Cluster order sorted by Stem Cell Differentiation score: ',list(cluster_order.index))
    display(cluster_order)
    return adata



def assess_start_cluster_plot(adata):
    """ Plot the G2M score and Stem Cell Differentiation score of each cluster

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
    
    ax1 = plt.subplot(211)
    ax1=sns.boxplot(adata.uns['G2M score'],x='cluster',y='G2M_score',linewidth=0.8,
                    palette=color_palette,
                   order=adata.uns['G2M score order'].index)

    ax1.set_xticklabels(adata.uns['G2M score order'].index.values,rotation=40,ha='center',fontsize=15)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('G2M score',fontsize=20)
    
    ax2 = plt.subplot(212)
    ax2=sns.boxplot(data=adata.uns['Stem Cell Differentiation score'],x='cluster',y='score',showfliers=False,
                    palette=color_palette,
                order=adata.uns['Stem Cell Differentiation score order'].index)
    ax2.set_xticklabels(adata.uns['Stem Cell Differentiation score order'].index.values,rotation=30,fontsize=15,ha='center')
    ax2.set_ylabel(' Stem Cell Differentiation score',fontsize=20)
    ax2.set_xlabel(' ')
    plt.tight_layout()

    
    
        
    
    
        
