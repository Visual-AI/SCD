import numpy as np
import copy
import random
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
import torch

## import for constrained km
from sklearn.utils.extmath import cartesian
import pyximport
pyximport.install()
from k_means_constrained.mincostflow_vectorized import SimpleMinCostFlowVectorized


class K_Means:
    def __init__(self, k =3, tolerance = 1e-4, max_iterations = 100, size_min = 100, size_max=1000, init='k-means++', n_init=10, random_state=None, n_jobs=None, pairwise_batch_size = None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.size_min = size_min
        self.size_max = size_max

    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)
        if(pre_centers is not None):
            C = pre_centers
        else:
            C = X[random_state.randint(0, len(X))]
        C = C.view(-1, X.shape[1])
        while C.shape[0]<k:
            dist = pairwise_distance(X, C, self.pairwise_batch_size) 
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1) 
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()
            ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)
        return C 


    def fit_once(self, X, random_state):
        centers = torch.zeros(self.k, X.shape[1]).type_as(X) 
        labels = -torch.ones(len(X)) 
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers 
        if self.init=='k-means++':
            centers= self.kpp(X, k=self.k, random_state=random_state) 
        elif self.init=='random':
            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False) 
            for i in range(self.k):
                centers[i] = X[idx[i]]
        else:
            for i in range(self.k):
                centers[i] = X[i]

        #begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):
            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            labels, inertia = _labels_constrained(X.cpu().numpy(), centers.cpu().numpy(), torch.sqrt(dist).cpu().numpy(), size_min=self.size_min, size_max=self.size_max, distances=np.zeros(shape=(X.shape[0],), dtype=X.cpu().numpy().dtype))
            labels = torch.from_numpy(labels)
            # mindist, labels = torch.min(dist, dim=1) 
            # inertia = mindist.sum()
            for idx in range(self.k):
                selected = torch.nonzero(labels==idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0) 
            
            if best_inertia is None or inertia < best_inertia: 
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift **2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break
        return best_labels, best_inertia, best_centers, i + 1

    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):
        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)
        l_classes = torch.unique(l_targets) 
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers= torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats) 
        centers[:len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long() 

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers 
        centers= self.kpp(u_feats, l_centers, k=self.k, random_state=random_state) 

        #begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()

            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_labels, u_inertia = _labels_constrained(u_feats.cpu().numpy(), centers.cpu().numpy(), torch.sqrt(dist).cpu().numpy(), size_min=self.size_min, size_max=self.size_max, distances=np.zeros(shape=(u_feats.shape[0],), dtype=u_feats.cpu().numpy().dtype))
            u_labels = torch.from_numpy(u_labels).type_as(labels)
            # u_mindist, u_labels = torch.min(dist, dim=1) 
            # u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum() 
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels  

            for idx in range(self.k):
                selected = torch.nonzero(labels==idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0) 

            if best_inertia is None or inertia < best_inertia: 
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift **2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break
        return best_labels, best_inertia, best_centers, i + 1

    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]

    def fit_mix(self, u_feats, l_feats, l_targets):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_mix_once(u_feats, l_feats, l_targets, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters 
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_mix_once)(u_feats, l_feats, l_targets, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)
    
    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2 
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis

def _labels_constrained(X, centers, D_sqrt, size_min, size_max, distances):
    """Compute labels using the min and max cluster size constraint

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.

    size_min : int
        Minimum size for each cluster

    size_max : int
        Maximum size for each cluster

    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    """
    C = centers

    # Distances to each centre C. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    # D = euclidean_distances(X, C, squared=False)
    D=D_sqrt

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, C, D, size_min, size_max)
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # Change distances in-place
    distances[:] = D[np.arange(D.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()

    return labels, inertia


def minimum_cost_flow_problem_graph(X, C, D, size_min, size_max):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # X-nodes: [0, n(x)-1], C-nodes: [n(X), n(X)+n(C)-1], C-dummy nodes:[n(X)+n(C), n(X)+2*n(C)-1],
    # Artificial node: [n(X)+2*n(C), n(X)+2*n(C)+1-1]

    # Create indices of nodes
    n_X = X.shape[0]
    n_C = C.shape[0]
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Edges
    edges_X_C_dummy = cartesian([X_ix, C_dummy_ix])  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs
    costs_X_C_dummy = D.reshape(D.size)
    costs = np.concatenate([costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))])

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C)
    ])

    # Sources and sinks
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate([
        supplies_X,
        np.zeros(n_C),  # C_dummies
        supplies_C,
        [supplies_art]
    ])

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype('int32')
    costs = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give extra precision
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies, n_C, n_X


def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlowVectorized()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    # Add each edge with associated capacities and cost
    min_cost_flow.AddArcWithCapacityAndUnitCostVectorized(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    min_cost_flow.SetNodeSupplyVectorized(np.arange(N_nodes, dtype='int32'), supplies)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.Solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_M = min_cost_flow.FlowVectorized(np.arange(n_X * n_C, dtype='int32')).reshape(n_X, n_C)

    labels = labels_M.argmax(axis=1)
    return labels

def main():
    import matplotlib.pyplot as plt
    from matplotlib import style
    import pandas as pd 
    style.use('ggplot')
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    from collections import Counter
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #  X = torch.from_numpy(X).float().to(device)


    y = np.array(y)
    l_targets = y[y>1]
    l_feats = X[y>1]
    u_feats = X[y<2]
    cat_feats = np.concatenate((l_feats, u_feats))
    y = np.concatenate((y[y>1], y[y<2]))
    cat_feats = torch.from_numpy(cat_feats).to(device)
    u_feats = torch.from_numpy(u_feats).to(device)
    l_feats = torch.from_numpy(l_feats).to(device)
    l_targets = torch.from_numpy(l_targets).to(device)

    km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10)

    #  km.fit(X)

    km.fit_mix(u_feats, l_feats, l_targets)
    #  X = X.cpu()
    X = cat_feats.cpu()
    centers = km.cluster_centers_.cpu()
    pred = km.labels_.cpu()
    print('nmi', nmi_score(pred, y))

    print(Counter(pred.numpy()))

    # Plotting starts here
    colors = 10*["g", "c", "b", "k", "r", "m"]

    for i in range(len(X)):
        x = X[i]
        plt.scatter(x[0], x[1], color = colors[pred[i]],s = 10)
 
    for i in range(4):
        plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')
    plt.show()

if __name__ == "__main__":
	main()
