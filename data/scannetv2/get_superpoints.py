import numpy as np
import numpy.matlib
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

import libcp
import libply_c


# https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/graphs.py
def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]

        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))

        edg_id = graph["source"] + n_ver * graph["target"]

        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]

        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    return graph, target2


# https://github.com/loicland/superpoint_graph/blob/2680a4dfbf173f1e4b4858112e25ab57fcd46907/partition/partition.py
def get_superpoints(xyz):
    """
    Assign each point to a superpoint and return these indices.

        Parameters:
            xyz: np.ndarray
                xyz coordinates of the point cloud being preprocessed, shape (n,3)

        Returns:
            superpoints: np.ndarray
                each xyz point's superpoint index, shape (n,)

    Adapted from:
        Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning,
        Loic Landrieu and Mohamed Boussaha CVPR, 2019.
    Code source:
        https://github.com/loicland/superpoint_graph
        partition/partition.py
    """
    k_nn_geof = 45
    k_nn_adj = 10
    lambda_edge_weight = 1.
    reg_strength = 0.1
    # d_se_max = 0
    # voxel_width = 0.03
    voxel_width = 0  # keep size of the point cloud the same

    if voxel_width > 0:
        xyz_ret = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0, 0)[0]
    else:
        xyz_ret = xyz

    #---compute 10 nn graph-------
    graph_nn, target_fea = compute_graph_nn_2(xyz_ret, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    geof = libply_c.compute_geof(xyz_ret, target_fea, k_nn_geof).astype('float32')

    del target_fea
    features = geof
    geof[:,3] = 2. * geof[:, 3]

    graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
    components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                    , graph_nn["edge_weight"], reg_strength)
    components = np.array(components, dtype = 'object')

    superpoints = np.hstack((
        np.vstack([np.ones((len(c), 1)) * i for i, c in enumerate(components)]),  # superpoint idx
        np.vstack([np.array(c).reshape((-1,1)) for c in components])              # point idx
    ))
    superpoints = superpoints[superpoints[:,1].argsort()]
    return superpoints[:,0]
