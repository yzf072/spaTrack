from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np

def nearest_neighbors(coord, coords, n_neighbors=5):
    neigh=NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(coords)
    _,neighs=neigh.kneighbors(np.atleast_2d(coord))
    return neighs

def kmeans_centers(coords, n_clusters=2):
    cell_coordinates=coords
    kmeans = KMeans(n_clusters)
    kmeans.fit(cell_coordinates)
    cluster_centers = kmeans.cluster_centers_
    print('kmean cluster centers:')
    list(map(print, cluster_centers))
    return cluster_centers

def dbscan_centers(coords,eps,min_samples):
    pass

