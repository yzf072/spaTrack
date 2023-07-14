
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random
import matplotlib.pyplot as plt
import sys
from scipy.sparse import issparse
import networkx as nx 
import matplotlib.pyplot as plt 

from typing import Literal, Union, List


class Model(nn.Module):
    """
    Model for exploring the relationship between TFs and genes.

    Parameters
    ----------
    n_gene
        The dimensionality of the input, i.e. the number of genes.
    n_TF
        The dimensonality of the output, i.e. the number of TFs.
    """
    def __init__(
        self,
        n_gene: int,
        n_TF: int,
    ) -> None:
        super(Model,self).__init__()
        self.n_gene=n_gene
        self.n_TF=n_TF

        self.linear=nn.Linear(n_gene,n_TF)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        Give the gene changes of cells, and get the relationship between genes and TFs through linear regression.

        Parameters
        ----------
        x
            The input data (gene changes)

        Returns
        -------
        :class:`torch.Tensor`

                    Tensors for the output data (TF expresssion):
        """
        y_pred = self.linear(x)
        return y_pred


class Trainer():
    """
    Class for implementing the training process.

    parameters
    ----------
    expression_matrix_path
        The path of the expression matrix file.
    ptime_path
        The path of the ptime file, used to determine the sequence of the ptime data.
    tfs_path
        The path of the tf names file.
    cell_mapping_path
        The path of the cell mapping file, where column `index_x` indicates the start cell and column `index_y` indicates the end cell.
    min_cells, optional
        The minimum number of cells for gene filtration, by default 100.
    cell_num_each_time, optional
        The cell number generated at each time point using the meta-analysis method, by default 500.
    random_select_cell_num, optional
        The number of random cells to use when generating cells, by default 10
    use_gpu, optional
        Whether to use gpu, by default True.
    """
    def __init__(
        self,
        type: Literal['2_time','p_time'],
        expression_matrix_path: Union[str, List[str]],
        tfs_path: str,
        cell_mapping_path: str = None,
        ptime_path: str = None,
        min_cells: Union[int,List[int]] = 100,
        cell_num_each_time: int = 500,
        random_select_cell_num: int = 10,
        sample_num: int =10000,
        cell_num: int =10,
        use_gpu: bool = True,
    ) -> None:
        # read gene expression, cell ptime and tfs files. 
        if type=='p_time':
            self.adata=sc.read(expression_matrix_path)
            self.adata.obs['ptime']=pd.read_table(ptime_path,index_col=0)
            self.tfs_path = tfs_path

            # filter out genes expressed in few cells
            sc.pp.filter_genes(self.adata, min_cells=min_cells,inplace=True)

            all_tfs=pd.read_csv(self.tfs_path,index_col=0)
            self.genes=self.adata.var_names
            self.tfs=self.genes.str.lower().intersection(all_tfs.gene.str.lower().tolist())

            self.generate_meta_data_p_time(cell_num_each_time,random_select_cell_num)
        elif type=='2_time':
            # 2_time Pattern need two adata files, the cell mapping file and the tfs file.
            self.adata1=sc.read(expression_matrix_path[0])
            self.adata2=sc.read(expression_matrix_path[1])
            self.cell_mapping=pd.read_csv(cell_mapping_path,index_col=0)
            self.tfs_path = tfs_path

            # filter genes
            sc.pp.filter_genes(self.adata1, min_cells=min_cells[0],inplace=True)
            sc.pp.filter_genes(self.adata2, min_cells=min_cells[1],inplace=True)

            self.generate_meta_data_2_time(sample_num,cell_num)

        gpu=torch.cuda.is_available() and use_gpu
        if gpu:
            self.device=torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def getMetaData(self,sample_num,cell_num):
        new_data_in=[]
        new_data_out=[]
        for i in range(sample_num):
            cell_indexes=random.sample(range(self.input_data.shape[0]),cell_num)
            new_data_in.append(np.sum(self.input_data[cell_indexes],axis=0))
            new_data_out.append(np.sum(self.output_data[cell_indexes],axis=0))
        
        self.input_data=np.stack(new_data_in)
        self.output_data=np.stack(new_data_out)


    def run(
        self,
        training_times: int = 10, 
        iter_times: int = 30, 
        mapping_num: int = 3000,
        filename='afterScale_100.csv',
    ) -> None:
        self.mapping_num=mapping_num
        self.all_gtf = []
        for i in range(training_times): 
            self.model=Model(len(self.genes),len(self.tfs)).to(self.device)
            loss_fn=nn.MSELoss()
            optimizer=torch.optim.SGD(self.model.parameters(),lr=0.1)

            epochs = iter_times
            for t in range(epochs): 
                print(f"Epoch {t+1}\n-------------------------------")
                self.train(self.train_dl, self.model, loss_fn, optimizer)
                self.test(self.test_dl, self.model, loss_fn)
            print("Done!")
            gtf=self.model.linear.weight.T
            self.all_gtf.append(gtf)

        # set the highest weighted map (TF itself) to 0
        sum_all_gtf=torch.sum(torch.stack(self.all_gtf),dim=0)
        _,self_idx=torch.max(sum_all_gtf,dim=0)
        for i in range(len(self_idx)):
            sum_all_gtf[self_idx[i],i]=0

        # save most important maps
        flat_tensor=torch.flatten(sum_all_gtf)
        sorted_tensor, _ = torch.sort(flat_tensor, descending=True)
        max_value=sorted_tensor[mapping_num]
        min_value=sorted_tensor[-(mapping_num+1)]

        self.max_TF_idx=torch.nonzero(sum_all_gtf>max_value)
        self.min_TF_idx=torch.nonzero(sum_all_gtf<min_value)

        network_rows=[]
        for i in self.max_TF_idx:
            gene=self.genes[i[0].item()]
            TF=self.tfs[i[1].item()]
            weight=sum_all_gtf[i[0].item(),i[1].item()].item()
            one_row=[TF,gene,weight]
            network_rows.append(one_row)
        for i in self.min_TF_idx:
            gene=self.genes[i[0].item()]
            TF=self.tfs[i[1].item()]
            weight=sum_all_gtf[i[0].item(),i[1].item()].item()
            one_row=[TF,gene,weight]
            network_rows.append(one_row)

        columns=['TF','gene','weight']
        self.network_df=pd.DataFrame(data=network_rows,columns=columns)
        self.network_df.to_csv(filename,index=0)
        print(f'weight relationships of tfs and genes are stored in {filename}')


    def plot_scatter(self,type='raise',num_rows=2,num_cols=5,fig_width=20,fig_height=8,):
        if self.mapping_num<(num_rows*num_cols):
            sys.exit('Mapping_num is less than the product of num_rows and num_cols, please run again and increase the mapping_num.')
        fig, axs = plt.subplots(num_rows, num_cols,figsize=(fig_width,fig_height),)
        for i, ax in enumerate(axs.flatten()):
            if type=='raise':
                gene=self.max_TF_idx[i][0]
                TF=self.max_TF_idx[i][1]
            elif type=='drop':
                gene=self.min_TF_idx[i][0]
                TF=self.min_TF_idx[i][1]
            x=self.output_data[:,TF]
            y=self.input_data[:,gene]
            
            ax.scatter(x, y, s=1, color='#377eb8')
            ax.set_title(f"{self.tfs[TF.item()]}, {self.genes[gene.item()]}")
            # ax.set_aspect('equal')
        # 展示图形
        # plt.tight_layout()
        # fig.subplots_adjust(left=0.06)
        plt.show()


    def generate_meta_data_2_time(self,sample_num,cell_num):
        # only use the mapping cells
        self.adata1=self.adata1[self.cell_mapping.index_x]
        self.adata2=self.adata2[self.cell_mapping.index_y]
        
        # get same genes
        same_genes = list(self.adata1.var_names & self.adata2.var_names)
        self.adata1 = self.adata1[:,same_genes]
        self.adata2 = self.adata2[:,same_genes]
        self.genes = self.adata1.var_names

        delta_gene = self.adata2.X-self.adata1.X
        if issparse(delta_gene):
            delta_gene=delta_gene.A

        # get tfs
        all_tfs=pd.read_csv(self.tfs_path,index_col=0)
        self.tfs=self.genes.str.lower().intersection(all_tfs.gene.str.lower().tolist())

        self.get_one_hot()

        self.input_data = np.array(delta_gene,dtype=np.float32)
        self.output_data = np.array(self.adata2.X @ self.T.T,dtype=np.float32)

        self.getMetaData(sample_num,cell_num)

        # normalize data 
        nor_input_data=(self.input_data-self.input_data.mean(axis=0))/(self.input_data.max(axis=0)-self.input_data.min(axis=0))
        nor_output_data=(self.output_data-self.output_data.min(axis=0))/(self.output_data.max(axis=0)-self.output_data.min(axis=0))

        # # shuffle the cell order (2_time data has been shuffled in the meta step)
        # permuted_idxs=np.random.permutation(self.input_data.shape[0])
        # self.input_data=nor_input_data[permuted_idxs]
        # self.output_data=nor_output_data[permuted_idxs]

        train_ratio=0.8
        num_samples=self.input_data.shape[0]
        train_size=int(num_samples*train_ratio)

        train_data_in=torch.from_numpy(nor_input_data[:train_size,:])
        train_data_out=torch.from_numpy(nor_output_data[:train_size,:])
        test_data_in=torch.from_numpy(nor_input_data[train_size:,:])
        test_data_out=torch.from_numpy(nor_output_data[train_size:,:]) 

        train_set=TensorDataset(train_data_in,train_data_out)
        test_set=TensorDataset(test_data_in,test_data_out)
        batch_size=32
        self.train_dl=DataLoader(train_set,batch_size=batch_size,shuffle=True)
        self.test_dl=DataLoader(test_set,batch_size=batch_size)


    def generate_meta_data_p_time(self,cell_num_each_time,random_select_cell_num) -> None:
        sub_index=self.sort_idx()

        mean_data=[]
        origin_data=[]  # time_point * cell * gene expression
        for i in range(len(sub_index)):
            origin_data.append(np.array(self.adata[sub_index[i]].X))
            mean_data.append(self.adata[sub_index[i]].X.mean(axis=0))
        mean_data=np.array(mean_data)
        origin_data=np.stack(origin_data)

        self.get_one_hot()

        input_data=[]
        output_data=[]
        for i in range(1,len(origin_data)):
            for j in range(cell_num_each_time):
                random_idxs=random.sample(range(len(origin_data[i])),random_select_cell_num)
                meta_expr=np.array(origin_data[i][random_idxs].mean(axis=0))
                delta_gene=meta_expr-mean_data[i-1]
                tf_expr=(origin_data[i][random_idxs].mean(axis=0))@self.T.T
                
                input_data.append(delta_gene)
                output_data.append(tf_expr)
        input_data=np.array(input_data,dtype=np.float32)
        output_data=np.array(output_data,dtype=np.float32)

        nor_input_data=(input_data-input_data.mean(axis=0))/(input_data.max(axis=0)-input_data.min(axis=0))
        nor_output_data=(output_data-output_data.min(axis=0))/(output_data.max(axis=0)-output_data.min(axis=0))

        permuted_idxs=np.random.permutation(input_data.shape[0])
        self.input_data=nor_input_data[permuted_idxs]
        self.output_data=nor_output_data[permuted_idxs]

        train_ratio=0.8
        num_samples=self.input_data.shape[0]
        train_size=int(num_samples*train_ratio)

        train_data_in=torch.from_numpy(self.input_data[:train_size,:])
        train_data_out=torch.from_numpy(self.output_data[:train_size,:])
        test_data_in=torch.from_numpy(self.input_data[train_size:,:])
        test_data_out=torch.from_numpy(self.output_data[train_size:,:]) 

        train_set=TensorDataset(train_data_in,train_data_out)
        test_set=TensorDataset(test_data_in,test_data_out)
        batch_size=32
        self.train_dl=DataLoader(train_set,batch_size=batch_size,shuffle=True)
        self.test_dl=DataLoader(test_set,batch_size=batch_size)


    def plot_gene_regulation(self,min_weight,min_node_num,cmap='coolwarm'):
        df=self.network_df
        df=df.loc[df['weight'].abs()>min_weight]
        print(f'num of weight pairs after weight filtering: {len(df)}')

        df_TF=pd.Series(df['TF'].value_counts())
        label_name=pd.Series(df_TF[df_TF>min_node_num].index).to_dict()
        df=df.loc[df['TF'].isin(label_name.values())]
        print(f'num of weight pairs after node_count filtering: {len(df)}')

        G = nx.from_pandas_edgelist(df, 'TF', 'gene', create_using = nx.Graph())

        nodes = G.nodes()
        degree = G.degree()
        colors = [degree[n] for n in nodes]
        #size = [(degree[n]) for n in nodes]

        pos = nx.kamada_kawai_layout(G)

        vmin = min(colors)
        vmax = max(colors)

        betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        node_color = [2000.0 * G.degree(v) for v in G]
        #node_color = [community_index[n] for n in H]
        node_size =  [v * 3000 for v in betCent.values()]

        label_name_new = dict(zip(label_name.values(), label_name.values()))


        fig = plt.figure(figsize = (8,8), dpi=200)
        nx.draw_networkx(G,pos,alpha = 0.8, node_color = node_color,
            node_size = node_size ,font_size = 20, width = 0.4, cmap = cmap,
                        with_labels=True, labels=label_name_new,edge_color ='grey')

    def get_one_hot(self,) -> None:
        """
        Generate one-hot matrix from TFs to genes.

        Returns
        -------
            one-hot matrix from TFs to genes
        """
        vectorizer=CountVectorizer(vocabulary=self.genes.str.lower().tolist())   # lowercase gene names
        self.T = vectorizer.fit_transform(self.tfs).toarray()


    def sort_idx(self):
        """
        _summary_

        Returns
        -------
            _description_
        """
        ptime=self.adata.obs['ptime']

        ptime_0_index=ptime[ptime==0].index
        ptime_1_index=ptime[ptime==1].index

        middle_idx=list(self.adata.obs.index[len(ptime_0_index):len(self.adata)-len(ptime_1_index)])
        sub_length=40
        sub_index=[middle_idx[i:i+sub_length] for i in range(0, len(middle_idx), sub_length)][:-1]

        # sub_index.insert(0,list(ptime_0_index)) # insert head indexes
        # sub_index.append(list(ptime_1_index))   # append tail inedexes

        return sub_index


    def train(self,dataloader, model, loss_fn, optimizer):
        size=len(dataloader.dataset)
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # 计算预测误差
            pred=model(X)
            loss=loss_fn(pred,y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 25 ==0:
                loss,current=loss.item(), (batch+1)*len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test(self,dataloader, model, loss_fn):
        # size=len(dataloader.dataset)
        num_batches=len(dataloader)
        model.eval()
        test_loss=0
        with torch.no_grad():
            for X,y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred=model(X)
                test_loss+=loss_fn(pred,y).item()
        test_loss/=num_batches
        print(f"Avg loss: {test_loss:>8f} \n")