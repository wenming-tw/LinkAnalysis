import os
import argparse
import numpy as np
from utils import edge_to_adjacency_matrix

def hits(adj, max_iter=100, tol=1e-8):
    N = adj.shape[0]

    hub = np.ones((N), dtype=np.float64)
    authority = np.ones((N), dtype=np.float64)
    
    for _ in range(max_iter):
        pre_authority = authority
        pre_hub = hub

        authority = adj.T.dot(pre_hub)
        hub = adj.dot(pre_authority)

        authority = authority / authority.sum()
        hub = hub / hub.sum()

        if (np.abs(authority - pre_authority) + np.abs(hub - pre_hub)).sum() < tol:
            break

    return authority, hub
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'graph_4.txt')
    parser.add_argument('--out_dir', default = '../results/')
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
        
    adj = edge_to_adjacency_matrix(edge, normalize=None)
    authority, hub = hits(adj, max_iter=args.max_iter)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(args.out_dir+'{}_authority.txt'.format(args.data.split('.')[0]), authority, delimiter=' ', fmt='%.6f')
    np.savetxt(args.out_dir+'{}_hub.txt'.format(args.data.split('.')[0]), hub, delimiter=' ', fmt='%.6f')



