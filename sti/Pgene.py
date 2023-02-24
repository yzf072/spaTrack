
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import stats
from sklearn import preprocessing
import  multiprocessing  as mp
from pygam import LinearGAM, s, f
import statsmodels.stats as stat
from scipy.signal import savgol_filter
import statsmodels.formula.api as smf
import  scipy.stats
import gc



"""
Caculate JS score to determine data trend is increase or decrease

"""

def js_score(gam_fit,grid_X ):
    """
    Parameters
    ----------
    gam_fit : 
            Fitted model by pyGAM

    grid_X: array
            An array value grided by pyGAM's generate_X_grid function 


    Returns
    -------
    trend: str
            Mark the fitted model is increase or decrease

    """

    def JS_divergence(p,q):
        """
        Parameters
        ----------
        p,q : 
                Two same length arrays
                p: array fitted by model
                q: standard distribution array

        Returns
        -------
            JS score. More smaller value indicate the distribution of inputed data is more similar with standard distribution
        """
        M=(p+q)/2
        return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

    x = [i for i in range(100)]
    l=np.array(x).reshape(-1,1)
    increase_trend=preprocessing.MaxAbsScaler().fit_transform(l).reshape(1,-1)[0]
    decrease_trend=increase_trend[::-1]
    decrease_score=JS_divergence( gam_fit.predict(grid_X),
                                 decrease_trend
                                )
    increase_score=JS_divergence( gam_fit.predict(grid_X),
                                 increase_trend
                                )
    pattern_dict = {'decrease':decrease_score,
                    'increase':increase_score}
    gene_trend = min(pattern_dict, 
                     key=pattern_dict.get
                    )
    return gene_trend


##Fit gene expression and ptime by generalized additive model
##Identify pesudotime-dependent genes may drive cell transition

"""
Filter genes by minimum expression proporation and cluster differential expression.
Cluster differential expression is used to as a reference to order gene.
"""

def filter_gene(adata,min_exp_prop,abs_FC):
    """
    Parameters
    ----------
    adata:
          scanpy adata for infering trajectory
    
    min_exp_prop: minimum expression proporation

    abs_FC: log2 |FC| in differential expression

    """

    ptime_list = list(adata.obs['ptime'])
    if sorted(ptime_list) == ptime_list:
        pass
    else:
        raise Exception ('error: Please sort adata by ptime')
        
    cluster_order = adata.obs.groupby(['cluster']).mean().sort_values(['ptime']).index

    ptime_sort_matrix = adata.X.copy()
    df_exp = pd.DataFrame(
        data=ptime_sort_matrix, 
        index=adata.obs.index, 
        columns=adata.var.index
    )
    
    #endog = adata.obs["ptime"]
    ##minimum expression proporation
    min_prop_filter=df_exp[df_exp.columns[(df_exp>0).sum(axis=0)>int(len(adata)*min_exp_prop)]]

    ##cluster differential expression
    sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df_diff_res = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals_adj','logfoldchanges']})
    diff_gene_list=list()

    print('The cluster order is:',end=' ')
    for cluster_name in cluster_order:
        print(cluster_name,end=' ')
        df_cluster_diff = df_diff_res.loc[df_diff_res[cluster_name+'_p']<0.01].sort_values([cluster_name+'_l'],ascending=False)
        gene_list1 = df_cluster_diff.loc[df_cluster_diff[cluster_name+'_l']>abs_FC][cluster_name+'_n']
        diff_gene_list = diff_gene_list+list(gene_list1)
    gene_list_lm=set(diff_gene_list).intersection(set(list(min_prop_filter.columns)))
    
    adata_filter = adata[:,min_prop_filter.columns]
    adata_filter.uns['gene_list_lm'] = gene_list_lm
    adata_filter.uns['diff_gene_list'] = diff_gene_list
    return adata_filter


"""
Function:

Called function  by  ptime_gene_GAM() for multi-process computing

"""
def GAM_gene_fit(exp_gene_list):
    
    """
    Parameters
    ----------
    exp_gene_list : multi layer list
    
    exp_gene_list[0]: dataframe
                    columns : ptime,gene_expression
    exp_gene_list[1]: gene_name

    """


    r_list = list()
    trend_list=list()
    gene_list=list()
    pvalue_list=list()

    df_new = exp_gene_list[0]
    gene = exp_gene_list[1]
    x = df_new[["ptime"]].values
    y = df_new[gene]
    gam = LinearGAM(s(0, n_splines=8))
    gam_fit=gam.gridsearch(x, y,progress=False)
    grid_X = gam_fit.generate_X_grid(term=0)
    r_list.append(gam_fit.statistics_['pseudo_r2']['explained_deviance'])
    pvalue_list.append(gam_fit.statistics_['p_values'][0])
    gene_list.append(gene)
            
    trend_list.append(
                js_score(gam_fit,grid_X )
              )

            ## sort gene by fdr and R2
    df_batch_res=pd.DataFrame({'gene':gene_list,
                             'pvalue':pvalue_list,
                             'model_fit':r_list,
                             'pattern':trend_list})
    return df_batch_res


"""
function:

perform GAM model fitted by ways of multi-process computing

"""

def ptime_gene_GAM(adata,
                   core_number = 3
                  ):

    """
    Parameters
    ----------
    adata : Anndata
            Scanpy adata for inferring trajectory.

    core_number : int
            Number of processes for caculating

    Returns
    -------
    df_res: dataframe
            pvalue: calculated from GAM
            R2: a goodness-of-fit measure. larger value means better fit
            pattern: increase or decrease. drection of gene expression changes across time
            fdr: BH fdr
            
    
    """
    ##perform GAM model on each gene
    #min_prop_filter,gene_list_for_gam,diff_gene_list = filter_gene(adata,min_exp_prop,abs_FC)
    gene_list_for_gam = adata.uns['gene_list_lm']
    
    df_exp_filter = pd.DataFrame(
        data = adata.X,
        index = adata.obs.index,
        columns = adata.var.index
    )

    print('The number of genes for GAM model: ',len(gene_list_for_gam))
    if core_number >=1:
        para_list=list()
        for gene in gene_list_for_gam:
            df_new=pd.DataFrame({'ptime':list(adata.obs["ptime"]),
                               gene:list(df_exp_filter[gene])})
            df_new=df_new.loc[df_new[gene]>0]
            para_list.append((df_new,gene))
        p = mp.Pool(core_number)
        df_res = p.map(GAM_gene_fit,para_list)
        p.close()
        p.join()
        df_res=pd.concat(df_res)
        
        del para_list
        gc.collect()
    fdr=stat.multitest.fdrcorrection(np.array(df_res['pvalue']))[1]
    df_res['fdr']=fdr
    df_res.index=list(df_res['gene'])
    return df_res

"""
function:

order gene by cluster's log2FC sorted by ptime

"""

def order_trajectory_genes(adata,df_sig_res):
    """
    Parameters
    ----------
    adata : AnnData object
          filtered adata 
          
            
    df_sig_res: dataframe
           return dataframe by ptime_gene_GAM() after filtering as significat gene dataframe
    
    Return
    ----------
    sig_gene_exp_order: dataframe 
            gene ordered expression dataframe for plotting heatmap 

    """ 
    #min_prop_filter,gene_list,diff_gene_list = filter_gene(adata,min_exp_prop)
    
    df_exp_filter = pd.DataFrame(
         data = adata.X,
         index = adata.obs.index,
         columns = adata.var.index
     )

    sig_gene_exp = df_exp_filter.loc[:, df_sig_res.index]
    ##sort gene by cluster differertial genes
    df1=pd.DataFrame ({'gene_name':adata.uns['diff_gene_list']})
    df2=pd.DataFrame({'gene_name':list(sig_gene_exp.columns)})
    ##sort df2 by df1 order
    df_merge=pd.merge(df1,df2,on=['gene_name'])
    
    order_list=list(df_merge.drop_duplicates()['gene_name'])
    sig_gene_exp_order = sig_gene_exp[order_list]
    
    return sig_gene_exp_order



def plot_trajectory_gene_heatmap(sig_gene_exp_order,
                 smooth_length,
                 #TF=False,
                 cmap_name= 'twilight_shifted'):
    """
    Parameters
    ----------
    sig_gene_exp_order : dataframe
            gene ordered expression dataframe.
    smooth_length: int 
             length of smoothing window 
    cmap_name : color palette

    Returns
    -------

    gg: fig
        Heatmap: column represents cells, index is genes

    """
        # z-score,normaliz data
    ## only show TF gene 
    #TF_file=pd.read_table('/hwfssz1/ST_SUPERCELLS/P21Z10200N0134/USER/huangke2/27.spatial.trajectory/09.sti.code/hs_hgnc_tfs.txt',header=None)
    #cell_TF_exp=cell_exp[cell_exp.columns[cell_exp.columns.isin(TF_file[0])]]

    sort_window_exog_z = stats.zscore(sig_gene_exp_order, axis=0)
    last_pd = pd.DataFrame(
        data=sort_window_exog_z.T, 
        columns=sort_window_exog_z.index, 
        index=sort_window_exog_z.columns
    )

    # smooth data
    last_pd_smooth = savgol_filter(last_pd ,
                                   smooth_length,
                                   1)
    last_pd_smooth=pd.DataFrame(last_pd_smooth)
    last_pd_smooth.columns=last_pd.columns
    last_pd_smooth.index=last_pd.index
    

    fig = plt.figure(figsize=(8,10))
    ax1 = plt.subplot2grid((8,10), (0,0), colspan=10, rowspan=8)
    pseudotime_gene_heatmap = sns.heatmap(last_pd_smooth,
                     cmap=cmap_name,
                       cbar_kws={'shrink': 0.3,'label': 'normalized expression'})
    cbar =  pseudotime_gene_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ## add cell type 
    #df_cell=pd.DataFrame(sig_gene_exp_order.index)
    #df_cell[1]=list(adata.obs['cluster'])
    #plt.axis('off')
    #cell_line_plot = sns.histplot(data = df_cell, x = 0,hue=1,ax=ax2)

    #cell_line_plot.set_frame_on(False)
    #cell_line_plot.get_legend().remove()
    #cell_line_plot._legend.remove()
    pseudotime_gene_heatmap.figure.axes[-1].yaxis.label.set_size(25)
    pseudotime_gene_heatmap.xaxis.tick_top()
    pseudotime_gene_heatmap.set_xticks([])
    pseudotime_gene_heatmap.yaxis.set_tick_params(labelsize=13)
    plt.xticks(rotation=90)
    return fig.tight_layout()



##plot example genes
def plot_trajectory_gene(adata,
                         gene_name,
                         line_width=5,
                         show_cell_type=False):
    """
    Parameters
    ----------
    adata : AnnData object.

    gene_name: str
          gene used to plot 
    line_width : int
          widthe of fitting line
    show_cell_type : bool
          whether to show cell type in plot

    Returns
    -------

    axs: fig object
        x axis indicate pseduotime; y axis indicate gene expression value

    """

    gene_expression = pd.DataFrame(
        data=adata.X, 
        index=adata.obs.index, 
        columns=adata.var.index
    )
    df_new=pd.DataFrame({'ptime':list(adata.obs["ptime"]),
                         gene_name:list(gene_expression[gene_name]),
                        'cell_type':list(adata.obs["cluster"])})
    df_new=df_new.loc[df_new[gene_name]>0]
    x_ptime = df_new[["ptime"]].values
    y_exp = df_new[gene_name]
    gam = LinearGAM(s(0, n_splines=10))
    gam_res=gam.gridsearch(x_ptime, 
                           y_exp,progress=False)
    
    fig, axs = plt.subplots(figsize=(10, 8))
    XX = gam_res.generate_X_grid(term=0)
    axs.plot(XX, 
            gam.predict(XX),
            color='#aa4d3d',
            linewidth=line_width)
    if show_cell_type==True:
        sns.scatterplot(x='ptime',
               y=gene_name, 
               palette='deep',
                ax=axs,
                data=df_new,
                hue='cell_type')
    else:
        sns.scatterplot(x='ptime',
            y=gene_name, 
           cmap="plasma",
            ax=axs,
            data=df_new,
            c=x_ptime)
    if show_cell_type==True:
        plt.gca().legend().set_title('')
        plt.legend(fontsize='xx-large',loc=(1.01, 0.5))
        
    plt.title(gene_name,
              fontsize=30)
    plt.xlabel('ptime',
               fontsize=30)
    plt.ylabel('expression',
               fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    return axs



def plot_trajectory_gene_list(adata,
                              gene_name_list,
                              col_num=4,
                              title_fontsize=25,
                              label_fontsize=22,
                              line_width=5,
                              fig_legnth=10,
                              fig_width=8):
    """
    Parameters
    ----------
    adata : AnnData object.

    gene_name_list: List object
             gene list used to plot
    col_num: int 
            Number of genes displayed per line in picature
            (Default: 4)
    title_fontsize: int
            title fontsize of picture
            (Default: 25)
    label_fontsize: int
             x and y label fontsize
            (Default: 22)
    fig_legnth,fig_width:
            The legenth and width of picture size
            (Default: 10,8)


    Returns
    -------

    ax: fig object
        x axis indicate pseduotime; y axis indicate gene expression value

    """

    gene_number=len(gene_name_list)
    row_num=math.ceil(gene_number/col_num)
    fig, axs = plt.subplots(ncols=col_num, 
                            nrows=row_num, 
                            figsize=(fig_legnth, fig_width),
                            sharey=False,
                            sharex=True)
    i=-1

    gene_expression = pd.DataFrame(
        data=adata.X, 
        index=adata.obs.index,
        columns=adata.var.index
    )

    for m in range(row_num):
        for n in range(col_num):
            i=i+1
            if i > gene_number-1:
                break
            gene_name=gene_name_list[i]

            ax=axs[m,n]
            df_new=pd.DataFrame({'ptime':list(adata.obs["ptime"]),
                                 gene_name:list(gene_expression[gene_name])}
                               )
            df_new=df_new.loc[df_new[gene_name]>0]
            x_ptime = df_new[["ptime"]].values
            y_exp = df_new[gene_name]
            gam = LinearGAM(s(0, 
                              n_splines=10))
            gam_res=gam.gridsearch(x_ptime,
                                   y_exp)
            XX = gam_res.generate_X_grid(term=0)
            ax.plot(XX, 
                    gam.predict(XX),
                    color='#aa4d3d',
                    linewidth=line_width)
            ax.scatter(x_ptime, y_exp, 
                       cmap="plasma",
                       c=x_ptime)
            ax.set_title(gene_name,
                         fontsize=title_fontsize)
    for pos in range(gene_number,row_num*col_num):
        axs.flat[pos].set_visible(False)
    
    #fig.tight_layout()
    fig.text(0.5, -0.04, 
             'ptime',
             ha='center',
             fontsize=label_fontsize)
    fig.text(0.01, 0.5, 
             'expression ', 
             va='center', 
             rotation='vertical',
             fontsize=label_fontsize)
    #plt.tight_layout()
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.06)
    return ax




##GAM 
##sig_gene_exp_order,df_res = ptime_gene_GAM(adata,min_exp_prop=0.2,mode_fit=0.2,FDR=0.05 )
##plot
##plot_trajectory_gene_heatmap(sig_gene_exp_order,
##                 smooth_length=600,
##                 #TF=False,
##                 cmap_name='twilight_shifted')
##                  #'seismic')
##plt.savefig('./heatmap.pdf')

#plot_trajectory_gene(adata,'MPO',show_cell_type=True)
#gene_list=['MPO','IL18','LYZ','ELP2']
#plot_trajectory_gene_list(adata,gene_list,col_num=2)
