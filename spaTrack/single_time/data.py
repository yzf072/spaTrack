import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
from typing import Optional
from matplotlib import cm, colors

# 细胞类型存储在adata.obs['cluster']中
# 细胞空间坐标存储在adata.obsm['spatial']中
# 细胞umap存储在adata.obsm['X_umap']中
all_data = [
    "TAM",
    "midbrain",
    "rongyuan",
    "moni3",
    "CAF",
    "endobrain",
    "neuro",
    "cortex",
]


def get_data(
    data_name: str,
):
    """
    Define and preprocess the test data

    Parameters
    ----------
    data_name
        Test data name

    Returns
    -------
        Anndata
    """
    if data_name == "TAM":
        print("TAM")
        adata = sc.read("./data/TAM/exp_2.txt", cache=True)
        annos = pd.read_csv("./data/TAM/anno_2.txt", sep="\t", header=None)
        adata.obs["cluster"] = annos[3].values

        sc.pp.filter_genes(adata, min_cells=3)  # 过滤基因
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "midbrain":
        print("空间组小鼠中脑数据")
        adata = ad.read("./data/midbrain/Dorsal_midbrain_cell_bin.h5ad")
        adata = adata[
            adata.obs["Batch"].isin(
                ["FP200000600TR_E3", "SS200000108BR_B1B2", "SS200000108BR_A3A4"]
            )
        ]
        adata = adata[
            ~adata.obs["annotation"].isin(["Fibro", "Micro", "Endo", "Ery"])
        ].copy()
        adata.obs.rename(columns={"annotation": "cluster"}, inplace=True)
        adata.X = adata.X.A
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "rongyuan":
        print("空间组蝾螈数据")
        adata = sc.read("./data/rongyuan/regionall_exp_count.txt", cache=True)
        coor = pd.read_table(
            "./data/rongyuan/regionall.coor.txt", header=None
        )  # 数据空间坐标
        annotation = pd.read_table(
            "./data/rongyuan/regionall.annotation.txt", header=None
        ).T
        cell_id = pd.read_table("./data/rongyuan/regionall.cell.id.txt", header=None)
        gene_id = pd.read_table("./data/rongyuan/regionall.gene.id.txt", header=None)
        adata.obs["cluster"] = annotation.values
        adata.obsm["spatial"] = np.array(coor)
        adata.obs.index = cell_id[0].values
        adata.var.index = gene_id[0].values
        # 蝾螈数据特定，反向一下x轴坐标
        adata.obsm["spatial"][:, 0] = -adata.obsm["spatial"][:, 0]

        cell_t1 = pd.read_table("./data/rongyuan/region1.cell.id.csv", header=None)[
            0
        ].values
        cell_t2 = pd.read_table("./data/rongyuan/region2.cell.id.csv", header=None)[
            0
        ].values
        cell_t3 = pd.read_table("./data/rongyuan/region3.cell.id.csv", header=None)[
            0
        ].values
        cell_t2 = np.setdiff1d(
            cell_t2, np.union1d(cell_t1, cell_t3), assume_unique=True
        )

        ind = adata.obs.index.isin(np.concatenate((cell_t1, cell_t2, cell_t3)))
        adata = adata[ind, :].copy()

        adata.obs.loc[cell_t1, "rtime"] = "t1"
        adata.obs.loc[cell_t2, "rtime"] = "t2"
        adata.obs.loc[cell_t3, "rtime"] = "t3"

        sc.pp.filter_genes(adata, min_cells=3)  # 过滤基因
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

        adata.uns["datatype"] = "spatial"
        return adata

    elif data_name == "moni3":
        print("第3次模拟数据 pca_umap")
        adata = sc.read("./data/moni3/exp.txt", cache=True)
        adata.obs["cluster"] = pd.read_csv("./data/moni3/ident.txt", header=None).values

        adata.obs.index=['cell_'+x for x in adata.obs.index]
        adata.var.index=['gene_'+x for x in adata.var.index]

        sc.pp.filter_genes(adata, min_cells=3)  # 过滤基因
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "CAF":
        print("CAF_偏大的一个数据集 标准化 pca_umap")
        adata = sc.read("./data/CAF/exp_2.txt", cache=True)
        annos = pd.read_csv("./data/CAF/anno_2.txt", sep="\t", header=None)
        adata.obs["cluster"] = annos[3].values
        sc.pp.filter_genes(adata, min_cells=3)  # 过滤基因
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "endobrain":
        print("小鼠内脑细胞 标准化 pca_umap")
        adata = sc.read(
            "./data/endobrain/Endothelial_raw_count_matrix_brain.txt", cache=True
        ).T
        info = pd.read_csv(
            "./data/endobrain/Endothelial_Metadata.csv", sep=";", index_col=0
        )
        cells = adata.obs_names.intersection(info.index)
        adata = adata[cells, :]
        adata.obs["cluster"] = info.loc[cells, "Cluster"].copy()
        adata.shape

        ind = ~adata.obs["cluster"].isin(
            ["choroid plexus", "artery shear stress", "interferon"]
        )
        adata = adata[ind, :]

        sc.pp.filter_genes(adata, min_cells=3)  # 过滤基因
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "neuro":
        print("齿状神经回细胞 20过滤 标准化 高变基因 pca_umap")
        info = pd.read_csv(
            "./data/neuro/GSE95315_10X_expression_data_v2.tab",
            index_col=0,
            sep="\t",
            nrows=2,
        )
        count = pd.read_csv(
            "./data/neuro/GSE95315_10X_expression_data_v2.tab",
            index_col=0,
            sep="\t",
            skiprows=[1, 2],
        ).T
        adata = sc.AnnData(X=count)
        adata.obs["cluster"] = info.loc["cluster_name"]

        ind = adata.obs["cluster"].isin(
            [
                "Granule-mature",
                "Granule-immature",
                "Neuroblast_2",
                "Neuroblast_1",
                "nIPC",
            ]
        )
        adata = adata[ind, :]

        adata.obs["cluster"] = adata.obs["cluster"].astype("category")
        adata.obs["cluster"].cat.reorder_categories(
            [
                "nIPC",
                "Neuroblast_1",
                "Neuroblast_2",
                "Granule-immature",
                "Granule-mature",
            ],
            inplace=True,
        )

        cols = list(map(colors.to_hex, cm.tab20.colors))
        adata.uns["cluster_colors"] = ["#aa40fc"] + [cols[i] for i in [3, 2, 1, 0]]

        sc.pp.filter_genes(adata, min_cells=20)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=2000, subset=True
        )

        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "cortex":
        print("人类皮质细胞 ")
        adata = sc.read("./data/cortex/EX_development_human_cortex_10X.h5ad")
        adata.obs.rename(columns={"celltype": "cluster"}, inplace=True)

        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)

        adata.uns["datatype"] = "sc"
        return adata

    elif data_name == "HSC":
        print("HSC")
        adata = sc.read("./data/HSC/exp_all.txt", cache=True)
        adata.obs.index = [str(i) for i in range(adata.n_obs)]
        adata = adata[:, 1:].copy()
        coor = pd.read_table("./data/HSC/coords.txt", header=None)
        annotation = pd.read_table("./data/HSC/ident.txt", header=None)
        annotation = annotation.replace(
            {
                0: "HSC",
                1: "GMP",
                2: "EEP",
                3: "Pro-B1",
                4: "Proerythroblast",
                5: "Myeloblast",
                6: "MKP",
                7: "BMEP",
                8: "Pro-B2",
                9: "CDP",
                10: "Pro-B2",
                11: "Pre-B",
                12: "Pro-B2",
                15: "EEP",
            }
        )
        gene_id = pd.read_table("./data/HSC/gene_all.txt", header=None)
        adata.obs["cluster"] = annotation[0].values
        adata.obsm["X_umap"] = np.array(coor)
        adata.var.index = np.array(gene_id[0])
        adata.uns["datatype"] = "sc"
        data_name = "HSC"
        return adata

    else:
        raise ValueError("Please give the right test data name.")
