import numpy as np

def edge_to_adjacency_matrix(edge, normalize=None):
    assert edge.shape[1] == 2

    N = edge.max() + 1
    adj = np.zeros((N, N), dtype=np.float64)
    adj[(edge.T[0], edge.T[1])] = 1.

    if normalize is not None:
        adj = adj / (adj.sum(axis=normalize, keepdims=True) + 1e-9)

    return adj