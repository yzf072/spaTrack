import ot
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.stats import norm
from numpy.random import RandomState
import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import VBox
import seaborn as sns
from typing import Optional, Union, Literal, Tuple, List

from .utils import nearest_neighbors, kmeans_centers


def get_ot_matrix(
    adata: AnnData,
    data_type: str,
    alpha1: int = 1,
    alpha2: int = 1,
    random_state: Union[None, int, RandomState] = 0,
) -> np.ndarray:
    """
    Calculate transfer probabilities between cells.

    Using optimal transport theory based on gene expression and/or spatial location information.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    data_type
        The type of sequencing data.

        - ``'spatial'``: for the spatial transcriptome data.
        - ``'single-cell'``: for the single-cell sequencing data.

    alpha1
        The proportion of spatial location information.
        (Default: 1)
    alpha2
        The proportion of gene expression information.
        (Default: 1)
    random_state
        Different initial states for the pca.
        (Default: 0)

    Returns
    -------
    :class:`~numpy.ndarray`
        Cell transition probability matrix.
    """
    if "X_pca" not in adata.obsm:
        print("X_pca is not in adata.obsm, automatically do PCA first.")
        sc.tl.pca(adata, svd_solver="arpack", random_state=random_state)
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

    else:
        sys.exit(
            "Please give the right data type, choose from 'spatial' or 'single-cell'."
        )

    a, b = (
        np.ones((adata.n_obs,)) / adata.n_obs,
        np.ones((adata.n_obs,)) / adata.n_obs,
    )
    lambd = 1e-1
    Gs = np.array(ot.sinkhorn(a, b, M, lambd))

    return Gs


def set_start_cells(
    adata: AnnData,
    select_way: Literal["coordinates", "cell_type"],
    cell_type: Optional[str] = None,
    start_point: Optional[Tuple[int, int]] = None,
    basis: str = "spatial",
    split: bool = False,
    n_clusters: int = 2,
    n_neigh: int = 5,
) -> list:
    """
    Use coordinates or cell type to manually select starting cells.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    select_way
        Ways to select starting cells.

        (1) ``'cell_type'``: select by cell type.
        (2) ``'coordinates'``: select by coordinates.

    cell_type
        Restrict the cell type of starting cells.
        (Deafult: None)
    start_point
        The coordinates of the start point in 'coordinates' mode.
    basis
        The basis in `adata.obsm` to store position information.
    split
        Whether to split the specific type of cells into several small clusters according to cell density.
    n_clsuters
        The number of cluster centers after splitting.
    n_neigh
        The number of neighbors next to the start point/cluster center selected as the starting cell.

    Returns
    -------
    list
        The index number of selected starting cells.
    """
    if select_way == "coordinates":
        if start_point is None:
            raise ValueError(
                f"`start_point` must be specified in the 'coordinates' mode."
            )

        start_cells = nearest_neighbors(start_point, adata.obsm["X_" + basis], n_neigh)[
            0
        ]

        if cell_type is not None:
            type_cells = np.where(adata.obs["cluster"] == cell_type)[0]
            start_cells = set(start_cells).intersection(set(type_cells))

    elif select_way == "cell_type":
        if cell_type is None:
            raise ValueError("in 'cell_type' mode, `cell_type` cannot be None.")

        start_cells = np.where(adata.obs["cluster"] == cell_type)[0]

        if split == True:
            mask = adata.obs["cluster"] == cell_type
            cell_coords = adata.obsm["X_" + basis][mask]
            cluster_centers = kmeans_centers(cell_coords, n_clusters=n_clusters)

            select_cluster_coords = adata.obsm["X_" + basis].copy()
            select_cluster_coords[np.logical_not(mask)] = 1e10
            start_cells = nearest_neighbors(
                cluster_centers, select_cluster_coords, n_neigh
            ).flatten()
    else:
        raise ValueError(f"`select_way` must choose from 'coordinates' or 'cell_type'.")

    return list(start_cells)


def get_ptime(adata: AnnData, start_cells: list):
    """
    Get the cell pseudotime based on transition probabilities from initial cells.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    start_cells
        List of index numbers of starting cells.

    Returns
    -------
    :class:`~numpy.ndarray`
        Ptime correspongding to cells.
    """
    select_trans = adata.obsp["trans"][start_cells]
    cell_tran = np.sum(select_trans, axis=0)
    adata.obs["tran"] = cell_tran
    cell_tran_sort = list(np.argsort(cell_tran))
    cell_tran_sort = cell_tran_sort[::-1]

    ptime = pd.Series(dtype="float32", index=adata.obs.index)
    for i in range(adata.n_obs):
        ptime[cell_tran_sort[i]] = i / (adata.n_obs - 1)

    return ptime.values


def get_neigh_trans(
    adata: AnnData, basis: str, n_neigh_pos: int = 10, n_neigh_gene: int = 0
):
    """
    Get the transport neighbors from two ways, position and/or gene expression

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The basis used in visualizing the cell position.
    n_neigh_pos
        Number of neighbors based on cell positions such as spatial or umap coordinates.
        (Default: 10)
    n_neigh_gene
        Number of neighbors based on gene expression (PCA).
        (Default: 0)

    Returns
    -------
    :class:`~scipy.sparse._csr.csr_matrix`
        A sparse matrix composed of transition probabilities of selected neighbor cells.
    """
    if n_neigh_pos == 0 and n_neigh_gene == 0:
        raise ValueError(
            "the number of position neighbors and gene neighbors cannot be zero at the same time."
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
        if "X_pca" not in adata.obsm:
            print("X_pca is not in adata.obsm, automatically do PCA first.")
            sc.tl.pca(adata)
        sc.pp.neighbors(
            adata, use_rep="X_pca", key_added="X_pca", n_neighbors=n_neigh_gene
        )

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

    trans_neigh_csr = csr_matrix(
        (csr_data, indices, indptr), shape=(adata.n_obs, adata.n_obs)
    )

    return trans_neigh_csr


def get_velocity(
    adata: AnnData,
    basis: str,
    n_neigh_pos: int = 10,
    n_neigh_gene: int = 0,
    grid_num=50,
    smooth=0.5,
    density=1.0,
) -> tuple:
    """
    Get the velocity of each cell.

    The speed can be determined in terms of the cell location and/or gene expression.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The label of cell coordinates, for example, `umap` or `spatial`.
    n_neigh_pos
        Number of neighbors based on cell positions such as spatial or umap coordinates.
        (Default: 10)
    n_neigh_gene
        Number of neighbors based on gene expression.
        (Default: 0)

    Returns
    -------
    tuple
        The grid coordinates and cell velocities on each grid to draw the streamplot figure.
    """
    adata.obsp["trans_neigh_csr"] = get_neigh_trans(
        adata, basis, n_neigh_pos, n_neigh_gene
    )

    position = adata.obsm["X_" + basis]
    V = np.zeros(position.shape)  # 速度为2维

    for cell in range(adata.n_obs):  # 循环每个细胞
        cell_u = 0.0  # 初始化细胞速度
        cell_v = 0.0
        x1 = position[cell][0]  # 初始化细胞坐标
        y1 = position[cell][1]
        for neigh in adata.obsp["trans_neigh_csr"][cell].indices:  # 针对每个邻居
            p = adata.obsp["trans_neigh_csr"][cell, neigh]
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
    print(f"The velocity of cells store in 'velocity_{basis}'.")

    P_grid, V_grid = get_velocity_grid(
        adata,
        P=position,
        V=adata.obsm["velocity_" + basis],
        grid_num=grid_num,
        smooth=smooth,
        density=density,
    )
    return P_grid, V_grid


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
    """
    Select the start cluster with the largest sum of transfer probability

    Parameters
    ----------
    adata
        Anndata
    clusters, list
        Give clusters to find, by default None, each cluster will be traversed and calculated

    Returns
    -------
    str
        One cluster with maximum sum of transition probabilities
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


def get_velocity_grid(
    adata,
    P: np.ndarray,
    V: np.ndarray,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:
    """
    Convert cell velocity to grid velocity for streamline display

    The visualization of vector field borrows idea from scTour: https://github.com/LiQian-XC/sctour/blob/main/sctour.

    Parameters
    ----------
    P
        The position of cells.
    V
        The velocity of cells.
    smooth
        The factor for scale in Gaussian pdf.
        (Default: 0.5)
    density
        grid density
        (Default: 1.0)
    Returns
    ----------
    tuple
        The embedding and unitary displacement vectors in grid level.
    """
    grids = []
    for dim in range(P.shape[1]):
        m, M = np.min(P[:, dim]), np.max(P[:, dim])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num * density))
        grids.append(gr)

    meshes = np.meshgrid(*grids)
    P_grid = np.vstack([i.flat for i in meshes]).T

    n_neighbors = int(P.shape[0] / grid_num)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(P)
    dists, neighs = nn.kneighbors(P_grid)

    scale = np.mean([grid[1] - grid[0] for grid in grids]) * smooth
    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]

    P_grid = np.stack(grids)
    ns = P_grid.shape[1]
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid * V_grid).sum(0))
    min_mass = 1e-5
    min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
    cutoff = mass < min_mass

    V_grid[0][cutoff] = np.nan

    adata.uns["P_grid"] = P_grid
    adata.uns["V_grid"] = V_grid

    return P_grid, V_grid


class Lasso:
    """
    Lasso an region of interest (ROI) based on spatial cluster.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    """

    __sub_index = []
    sub_cells = []

    def __init__(self, adata):
        self.adata = adata

    def vi_plot(
        self,
        basis: str = "spatial",
        cell_type: Optional[str] = None,
    ):
        """
        Plot figures.

        Parameters
        ----------
        basis
            The basis in `adata.obsm` to store position information.
            (Deafult: 'spatial')
        cell_type
            Restrict the cell type of starting cells.
            (Deafult: None)

        Returns
        -------
            The container of cell scatter plot and table.
        """
        cell_types = self.adata.obs["cluster"].unique()
        colors = sns.color_palette(n_colors=len(cell_types)).as_hex()
        cluster_color = dict(zip(cell_types, colors))
        self.adata.uns["cluster_color"] = cluster_color

        df = pd.DataFrame()
        df["group_ID"] = self.adata.obs_names
        df["labels"] = self.adata.obs["cluster"].values
        df["spatial_0"] = self.adata.obsm["X_" + basis][:, 0]
        df["spatial_1"] = self.adata.obsm["X_" + basis][:, 1]
        df["color"] = df.labels.map(self.adata.uns["cluster_color"])

        py.init_notebook_mode()

        f = go.FigureWidget(
            [
                go.Scatter(
                    x=df["spatial_0"],
                    y=df["spatial_1"],
                    mode="markers",
                    marker_color=df["color"],
                )
            ]
        )
        scatter = f.data[0]
        f.layout.plot_bgcolor = "rgb(255,255,255)"
        f.layout.autosize = False

        axis_dict = dict(
            showticklabels=True,
            autorange=True,
        )
        f.layout.yaxis = axis_dict
        f.layout.xaxis = axis_dict
        f.layout.width = 600
        f.layout.height = 600

        # Create a table FigureWidget that updates on selection from points in the scatter plot of f
        t = go.FigureWidget(
            [
                go.Table(
                    header=dict(
                        values=["group_ID", "labels", "spatial_0", "spatial_1"],
                        fill=dict(color="#C2D4FF"),
                        align=["left"] * 5,
                    ),
                    cells=dict(
                        values=[
                            df[col]
                            for col in ["group_ID", "labels", "spatial_0", "spatial_1"]
                        ],
                        fill=dict(color="#F5F8FF"),
                        align=["left"] * 5,
                    ),
                )
            ]
        )

        def selection_fn(trace, points, selector):

            t.data[0].cells.values = [
                df.loc[points.point_inds][col]
                for col in ["group_ID", "labels", "spatial_0", "spatial_1"]
            ]

            Lasso.__sub_index = t.data[0].cells.values[0]
            Lasso.sub_cells = np.where(self.adata.obs.index.isin(Lasso.__sub_index))[0]

            if cell_type is not None:
                type_cells = np.where(self.adata.obs["cluster"] == cell_type)[0]
                Lasso.sub_cells = sorted(
                    set(Lasso.sub_cells).intersection(set(type_cells))
                )

        scatter.on_selection(selection_fn)

        # Put everything together
        return VBox((f, t))
