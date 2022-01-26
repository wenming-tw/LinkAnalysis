import os
import time
import argparse
import numpy as np
from utils import edge_to_adjacency_matrix

def pagerank(adj, d=0.15, max_iter=100, tol=1e-8):
    N = adj.shape[0]
    pg = np.ones((N)) / N

    for _ in range(max_iter):
        pre_pg = pg
        pg = d/N + (1-d)*(adj.T.dot(pg))
        
        if (np.abs(pg - pre_pg)).sum() < tol:
            break
        
    return pg / pg.sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'graph_4.txt')
    parser.add_argument('--out_dir', default = '../results/')
    parser.add_argument('--damping_factor', default = 0.15, type = float)
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

    adj = edge_to_adjacency_matrix(edge, normalize=1)
    pg = pagerank(adj, d=args.damping_factor, max_iter=args.max_iter)
    
    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(args.out_dir+'{}_PageRank.txt'.format(args.data.split('.')[0]), pg, delimiter=' ', fmt='%.6f')




