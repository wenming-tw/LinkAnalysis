import os
import argparse
import numpy as np
from utils import edge_to_adjacency_matrix

# https://en.wikipedia.org/wiki/SimRank, Matrix representation of SimRank
def simrank(adj, decay_factor=0.9, max_iter=100, tol=1e-8):
    N = adj.shape[0]
    
    diagonal_idx = (np.arange(N, dtype=int), np.arange(N, dtype=int))
    sim = np.zeros_like(adj)
    sim[diagonal_idx] = 1
    
    for _ in range(max_iter):
        pre_sim = sim
        sim = decay_factor*(adj.T.dot(sim).dot(adj))
        sim[diagonal_idx] = 1

        if (np.abs(sim - pre_sim)).sum() < tol:
            break

    return sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'graph_4.txt')
    parser.add_argument('--out_dir', default = '../results/')
    parser.add_argument('--decay_factor', default = 0.9, type = float)
    parser.add_argument('--max_iter', default = 100, type = int)
    args = parser.parse_args()

    if 'ibm' in args.data:
        edge = np.loadtxt('../data/'+args.data, dtype=int)[:,-2:]
        edge -= 1
        edge[:,1] += edge[:,0].max() + 1
        edge = np.concatenate([edge, np.flip(edge, 1)], axis=0) # bipartite graph
    else:
        edge = np.loadtxt('../data/'+args.data, delimiter=',', dtype=int)
        edge -= 1
        
    adj = edge_to_adjacency_matrix(edge, normalize=0)
    sim = simrank(adj, decay_factor=args.decay_factor, max_iter=args.max_iter)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(args.out_dir+'{}_SimRank.txt'.format(args.data.split('.')[0]), sim, delimiter=' ', fmt='%.6f')




