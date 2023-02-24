import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from typing_extensions import Literal
from sklearn.metrics.pairwise import euclidean_distances
import ot
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.stats import norm
from typing import Optional

from sti.utils import nearest_neighbors, kmeans_centers


def read_file(expr, coor, annotation, cell_id, gene_id):
    """read the expression matrix, coordinates, annotations,cell ids and gene ids

    Args:
        expr (str): path of the expression matrix
        coor (str): path of cell coordinates
        annotation (str): path of cell annotations, like cluster

    Returns:
        anndata: adata with X, cluster and coordinates
    """
    adata = sc.read(expr, cache=True)
    coor = pd.read_table(coor, header=None)
    annotation = pd.read_table(annotation, header=None).T
    cell_id = pd.read_table(cell_id, header=None)
    gene_id = pd.read_table(gene_id, header=None)

    adata.obs["cluster"] = annotation.values
    adata.obsm["spatial"] = np.array(coor)
    adata.obs.index = cell_id[0].values
    adata.var.index = gene_id[0].values

    return adata


def preprocess(adata, gene_in_min_cells=20):
    sc.pp.filter_genes(adata, min_cells=gene_in_min_cells)  # 过滤基因
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def plot(
    adata,
):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
    # fig.suptitle()
    sc.pl.umap(
        adata,
        color="spatial",
        size=20,
        ax=axs[0][0],
        legend_loc="on data",
        legene_fontoutline=3,
        show=False,
        s=50,
    )


def get_ot_matrix(
    adata: AnnData,
    data_type: Literal["spatial", "single-cell"] = "single-cell",
    alpha1: int = 1,
    alpha2: int = 1,
):
    if "X_pca" not in adata.obsm:
        print("X_pca is not in adata.obsm, automatically do PCA first.")
        sc.tl.pca(adata, svd_solver="arpack")
    newdata = adata.obsm["X_pca"]
    newdata2 = newdata.copy()

    if data_type == "spatial":
        newcoor = adata.obsm["X_spatial"]
        newcoor2 = newcoor.copy()

        # calculate physical distance
        ed_coor = euclidean_distances(newcoor, newcoor2, squared=True)
        m1 = ed_coor / sum(sum(ed_coor))
        # calculate gene expression PCA space distance
        ed_gene = euclidean_distances(newdata, newdata2, squared=True)
        m2 = ed_gene / sum(sum(ed_gene))

        M = alpha1 * m1 + alpha2 * m2
        M /= M.max()
        row, col = np.diag_indices_from(M)
        M[row, col] = M.max() * 1000000

    elif data_type == "single-cell":
        ed = euclidean_distances(newdata, newdata2, squared=True)
        M = ed / sum(sum(ed))
        M /= M.max()
        row, col = np.diag_indices_from(M)
        M[row, col] = M.max() * 1000000

    a, b = (
        np.ones((adata.n_obs,)) / adata.n_obs,
        np.ones((adata.n_obs,)) / adata.n_obs,
    )
    lambd = 1e-1
    Gs = ot.sinkhorn(a, b, M, lambd)

    return Gs


def select_cluster(
    adata, up=float("inf"), down=float("-inf"), left=float("-inf"), right=float("inf")
):
    """choose the original cells

    Args:
        adata (anndata): anndata with
        up, down, left, right (int): thresholds

    Returns:
        np.array: boolean array, length is the cell number
    """
    select = (
        (adata.obsm["spatial"][:, 0] > left)
        & (adata.obsm["spatial"][:, 0] < right)
        & (adata.obsm["spatial"][:, 1] < up)
        & (adata.obsm["spatial"][:, 1] > down)
    )
    return select


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def save_coordinate(df, selection, filename, sep="\t"):
    df_select = df.iloc[selection.index]
    df_select.to_csv(filename, sep="\t")


def save_results():
    option = input("Save your the results? Y/N:  ")
    if option == "Y":
        filename = input(
            "Please give the name of the coordiante file, e.g filename.csv:  "
        )
        save_coordinate(df, selection, filename=filename)
    else:
        print("The results will be discarded!")


def selected_info(index):
    # Write function that uses the selection indices to slice points and compute stats
    selected = points.iloc[index]
    if index:
        print(selected.array())
        #         label = 'Mean x, y: %.3f, %.3f' % tuple(selected.array().mean(axis=0))
        label = "Mean x, y: %.3f, %.3f" % tuple(
            selected.array()[:, [0, 1]].mean(axis=0)
        )  # 选则前两列计算x和y

    else:
        label = "No selection"
    type(selected.relabel(label))
    return selected.relabel(label).opts(color="color")


def set_start_cells(
    adata,
    select_way: Literal["cell_type", "coordinates", "partition"],
    cell_type=None,
    up=float("inf"),
    down=float("-inf"),
    left=float("-inf"),
    right=float("inf"),
    basis="spatial",
    n_neigh=5,
    cluster_centers=None,
    n_clusters=2,
):
    """Artificially set start cells

    Parameters
    ----------
    adata
        anndata
    select_way
        The way to select the starting cells, including cell type and coordinates
    cell_type, optional
        Give out the selected starting cell type when the `select_way` is 'cell_type', by default None
    up, optional
        Upper bound of coordinates, by default float('inf')
    down, optional
        Lower bound of coordinates, by default float('-inf')
    left, optional
        Left bound of coordinates, by default float('-inf')
    right, optional
        Right bound of coordinates, by default float('inf')

    Returns
    -------
        Give out the index of the selected starting  cells
    """
    if select_way == "number":
        select = (
            (adata.obsm["spatial"][:, 0] > left)
            & (adata.obsm["spatial"][:, 0] < right)
            & (adata.obsm["spatial"][:, 1] < up)
            & (adata.obsm["spatial"][:, 1] > down)
        )
        start_cells = np.where(select)[0].tolist()
    elif select_way == "cell_type":
        start_cells = np.where(adata.obs["cluster"] == cell_type)[0].tolist()
    elif select_way == "partition":
        if cell_type == None:
            mask = np.array([True] * adata.n_obs)
        else:
            mask = adata.obs["cluster"] == cell_type

        if cluster_centers is None:
            cell_coords = adata.obsm["X_" + basis][mask]
            cluster_centers = kmeans_centers(cell_coords, n_clusters=n_clusters)

        select_cluster_coords = adata.obsm["X_" + basis].copy()
        select_cluster_coords[np.logical_not(mask)] = 1e10
        start_cells = nearest_neighbors(
            cluster_centers, select_cluster_coords, n_neigh
        ).flatten()

    return start_cells
    # elif select_way == 'lasso':
    #     data = {
    #         'x_values': adata.obsm['spatial'][:, 0].tolist(),
    #         'y_values': adata.obsm['spatial'][:, 1].tolist(),
    #         'color': adata.obs['cluster'].tolist()
    #     }
    #     opts.defaults(
    #         opts.Points(tools=['box_select', 'lasso_select'],
    #                     nonselection_alpha=0.9))
    #     points = hv.Points(data,
    #                        kdims=['x_values', 'y_values'],
    #                        vdims=['color']).opts(color='color',
    #                                              width=700,
    #                                              cmap='Category20',
    #                                              height=700,
    #                                              size=4.5)
    #     selection = streams.Selection1D(source=points)

    #     # Combine points and DynamicMap
    #     return points,selection
    # else:
    #     raise KeyError(
    #         f"'{select_way}' is not valid for start cells selecting. Please choose from 'lasso' and 'number'."
    #     )


def get_ptime(adata, start_cells):
    """Get the cell ptime according to the choose of start cells

    Parameters
    ----------
    adata
        anndata
    start_cells
        Selected starting cells

    Returns
    -------
        pd.Series: Ptime correspongding to the cell
    """
    select_trans = adata.obsp["trans"][start_cells]
    adata.obs["tran"] = np.sum(select_trans, axis=0)
    cell_tran_sort = list(np.argsort(adata.obs["tran"]))
    cell_tran_sort = cell_tran_sort[::-1]

    ptime = pd.Series(dtype="float32", index=adata.obs.index)
    for i in range(adata.n_obs):
        ptime[cell_tran_sort[i]] = i / (adata.n_obs - 1)

    return ptime.values


def get_neigh_trans(adata, basis, n_neigh_pos=10, n_neigh_gene=10):
    """Get the transport neighbors from two ways, position and gene expression

    Parameters
    ----------
    adata
        Annadata
    basis
        The basis used in visualizing the cell position
    n_neigh_pos, optional
        Number of neighbors based on cell positions such as spatial or umap coordinates, by default 10
    n_neigh_gene, optional
        Number of neighbors based on gene expression (PCA), by default 10

    Returns
    -------
        Selected transport neighbors
    """
    if n_neigh_pos == 0 and n_neigh_gene == 0:
        raise ValueError(
            "the number of position neighbors and gene neighbors cannot be zero at the same time"
        )

    if n_neigh_pos:
        nn = NearestNeighbors(n_neighbors=n_neigh_pos, n_jobs=-1)
        nn.fit(adata.obsm["X_" + basis])
        dist_pos, neigh_pos = nn.kneighbors(adata.obsm["X_" + basis])
        dist_pos = dist_pos[:, 1:]
        neigh_pos = neigh_pos[:, 1:]

        neigh_pos_list = []
        for i in range(adata.n_obs):
            idx = neigh_pos[i]  # embedding上的邻居
            idx2 = neigh_pos[idx]  # embedding上邻居的邻居
            idx2 = np.setdiff1d(idx2, i)

            neigh_pos_list.append(np.unique(np.concatenate([idx, idx2])))
            # neigh_pos_list.append(idx)

    if n_neigh_gene:
        # if "pca" not in adata.uns.keys():
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neigh_gene, n_pcs=30, knn=True)

        neigh_gene = adata.obsp["distances"].indices.reshape(
            -1, adata.uns["neighbors"]["params"]["n_neighbors"] - 1
        )

    indptr = [0]
    indices = []
    csr_data = []
    count = 0
    for i in range(adata.n_obs):
        if n_neigh_pos == 0:
            n_all = neigh_gene[i]
        elif n_neigh_gene == 0:
            n_all = neigh_pos_list[i]
        else:
            n_all = np.unique(np.concatenate([neigh_pos_list[i], neigh_gene[i]]))
        count += len(n_all)
        indptr.append(count)
        indices.extend(n_all)
        csr_data.extend(
            adata.obsp["trans"][i][n_all]
            / (adata.obsp["trans"][i][n_all].sum())  # normalize
        )

    trans_neigh = csr_matrix(
        (csr_data, indices, indptr), shape=(adata.n_obs, adata.n_obs)
    )

    return trans_neigh, trans_neigh.A


def get_velocity(adata, basis, n_neigh_pos=10, n_neigh_gene=10):
    """Get the velocity of each cell. The speed can be determined in terms of cell location and gene expression

    Parameters
    ----------
    adata
        Anndata
    basis
        The label of cell coordinates, for example, `umap` or `spatial`
    n_neigh_pos, optional
        Number of neighbors based on cell positions such as spatial or umap coordinates, by default 10
    n_neigh_gene, optional
        Number of neighbors based on gene expression (PCA), by default 10

    Returns
    -------
        The cell velocity on each grid to draw the streamplot figure
    """
    adata.obsp["trans_neigh_csr"], adata.obsp["trans_neigh"] = get_neigh_trans(
        adata, basis, n_neigh_pos, n_neigh_gene
    )
    position = adata.obsm["X_" + basis]
    # 求出估计速度ff
    V = np.zeros(position.shape)  # 速度为2维

    for cell in range(adata.n_obs):  # 循环每个细胞
        cell_u = 0.0  # 初始化细胞速度
        cell_v = 0.0
        x1 = position[cell][0]  # 初始化细胞坐标
        y1 = position[cell][1]
        for neigh in adata.obsp["trans_neigh_csr"][cell].indices:  # 针对每个邻居
            p = adata.obsp["trans_neigh"][cell][neigh]
            if (
                adata.obs["ptime"][neigh] < adata.obs["ptime"][cell]
            ):  # 若邻居的ptime小于当前的，则概率反向
                p = -p

            x2 = position[neigh][0]
            y2 = position[neigh][1]

            # 正交向量确定速度方向，乘上概率确定速度大小
            sub_u = p * (x2 - x1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            sub_v = p * (y2 - y1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            cell_u += sub_u
            cell_v += sub_v
        V[cell][0] = cell_u / adata.obsp["trans_neigh_csr"][cell].indptr[1]
        V[cell][1] = cell_v / adata.obsp["trans_neigh_csr"][cell].indptr[1]
    adata.obsm["velocity_" + basis] = V

    E_grid, V_grid = vector_field_embedding_grid(
        adata, E=position, V=adata.obsm["velocity_" + basis]
    )
    return E_grid, V_grid


def get_2_biggest_clusters(adata):
    """Automatic selection of starting cells to determine the direction of cell trajectories

    Args:
        adata (anndata):

    Returns:
        tuple: Contains 2 cluster with maximum sum of transition probabilities
    """
    clusters = np.unique(adata.obs["cluster"])
    cluster_trans = pd.DataFrame(index=clusters, columns=clusters)
    for start_cluster in clusters:
        for end_cluster in clusters:
            if start_cluster == end_cluster:
                cluster_trans.loc[start_cluster, end_cluster] = 0
                continue
            starts = adata.obs["cluster"] == start_cluster
            ends = adata.obs["cluster"] == end_cluster
            cluster_trans.loc[start_cluster, end_cluster] = (
                np.sum(adata.obsp["trans"][starts][:, ends])
                / np.sum(starts)
                / np.sum(ends)
            )
    highest_2_clusters = cluster_trans.stack().astype(float).idxmax()
    return highest_2_clusters


def auto_get_start_cluster(adata, clusters: Optional[list] = None):
    """Select the start cluster with the largest sum of transfer probability

    Parameters
    ----------
    adata
        Anndata
    clusters, list
        Give clusters to find, by default None, each cluster will be traversed and calculated

    Returns
    -------
        string: One cluster with maximum sum of transition probabilities
    """
    if clusters == None:
        clusters = np.unique(adata.obs["cluster"])
    cluster_prob_sum = {}
    for cluster in clusters:
        start_cells = set_start_cells(adata, select_way="cell_type", cell_type=cluster)
        adata.obs["ptime"] = get_ptime(adata, start_cells)
        cell_time_sort = adata.obs["ptime"].values.argsort()

        prob_sum = 0
        for i in range(len(cell_time_sort) - 1):
            pre = cell_time_sort[i]
            next = cell_time_sort[i + 1]
            prob = adata.obsp["trans"][pre, next]
            prob_sum += prob
        cluster_prob_sum[cluster] = prob_sum
    highest_cluster = max(cluster_prob_sum, key=cluster_prob_sum.get)

    print(
        "The auto selecting cluster is: '"
        + highest_cluster
        + "'. If there is a large discrepancy with the known biological knowledge, please manually select the starting cluster."
    )

    return highest_cluster


def vector_field_embedding_grid(
    adata,
    E: np.ndarray,
    V: np.ndarray,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:
    """
    Estimate the unitary displacement vectors within a grid.
    This function borrows the ideas from scvelo: https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_grid.py.
    Parameters
    ----------
    E
        The embedding.
    V
        The unitary displacement vectors under the embedding.
    smooth
        The factor for scale in Gaussian pdf.
        (Default: 0.5)
    stream
        Whether to adjust for streamplot.
        (Default: `False`)
    density
        grid density
        (Default: 1.0)
    Returns
    ----------
    tuple
        The embedding and unitary displacement vectors in grid level.
    """

    grs = []
    for i in range(E.shape[1]):
        m, M = np.min(E[:, i]), np.max(E[:, i])  # 提取该维度上的最小值和最大值
        diff = M - m
        m = m - 0.01 * diff
        M = M + 0.01 * diff
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes = np.meshgrid(*grs)
    E_grid = np.vstack([i.flat for i in meshes]).T

    n_neigh = int(E.shape[0] / 50)
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
    nn.fit(E)
    dists, neighs = nn.kneighbors(E_grid)

    scale = np.mean([g[1] - g[0] for g in grs]) * smooth
    weight = norm.pdf(x=dists, scale=scale)
    weight_sum = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, weight_sum)[:, None]

    E_grid = np.stack(grs)
    ns = E_grid.shape[1]
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid * V_grid).sum(0))
    min_mass = 1e-5
    min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
    cutoff = mass < min_mass

    V_grid[0][cutoff] = np.nan

    adata.uns["E_grid"] = E_grid
    adata.uns["V_grid"] = V_grid

    return E_grid, V_grid
