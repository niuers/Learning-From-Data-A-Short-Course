import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import pandas as pd
import numpy as np
import math
from functools import partial
import collections

import sklearn
from scipy.spatial import distance

# Compute the Euler distance between two R^2 points
# x,z are 2D arrays
def dist(x, z, dist_type= 'euclidean'):
    res = distance.cdist(x, z, dist_type)
    return res

def build_distance_matrix(X):
    N, d = X.shape
    D = np.zeros((N, N))
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            D[i,j] = dist(X[i,:].reshape(-1, d), 
                          X[j, :].reshape(-1, d)).ravel()[0]
            D[j, i] = D[i, j]
    return D

def create_prototypes(X):
    # Assuming all points in X belong to one class, this function create a condensed point
    # for the points in X. It merges two closest points one by one
    #X: Nxd matrix
    Z = X 
    while Z.shape[0] > 1:
        N, d = Z.shape
        distances = []
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                x = Z[i, :].reshape(1, -1)
                y = Z[j, :].reshape(1, -1)
                dis = distance.cdist(x, y, 'euclidean')
                distances.append(dis.ravel()[0])
                pairs.append((i,j))
        #print(Z)
        #print(distances, pairs)
        order = np.argsort(np.array(distances))
        #print(order)
        idx1, idx2 = pairs[order[0]]
        #print(idx1, idx2)
        new_x = (Z[idx1, :]+Z[idx2,:])/2
        Z = np.delete(Z, ([idx1, idx2]), axis=0)
        Z = np.vstack([Z, np.array(new_x)])
    return Z

def find_nn_idx(x, X, k):
    """Find the first k nearest neighbors of x from points in X
    """
    # Find the indexes of k nearest neighbors for x
    distances = dist(x, X).ravel()
    order = np.argsort(np.array(distances))
    return order[:k], distances[order[:k]]

class NearestNeighbors:
    def __init__(self, X, y, k, problem_type='classification', transformer=None):
        #X: Nxd matrix, where each row corresponds to a data point x in R^d
        self.X = X 
        self.y = y 
        self.k = k #number of nearest neighbors
        self.problem_type = problem_type
        self.transformer = transformer

    def find_nn_idx(self, x, k):
        # Find the indexes of k nearest neighbors for x
        ret, _= find_nn_idx(x, self.X, k)
        return ret

    def find_nn(self, x):
        # Find the nearest neighbors for x
        distances = dist(x, self.X).ravel()
        order = np.argsort(np.array(distances))
        nns_x = self.X[order, :]
        nns_y = self.y[order]
        return nns_x[:self.k], nns_y[:self.k]

    def predict_one(self, x):
        # Predict one input point
        _, nns_y = self.find_nn(x)
        y = np.sum(nns_y)

        if self.problem_type == 'classification':
            y = np.sign(y)
        elif self.problem_type == 'regression':
            y = y / len(nns_y)
        else:
            raise ValueError("Non-recognizable problem type")
        
        return y

    def predict(self, X):
        # Predict the y for input X: Mxd matrix
        if self.transformer is not None:
            X = self.transformer(X)
            
        M, _ = X.shape
        predicted = []
        for idx in np.arange(M):
            x = X[idx, :].reshape(1, -1)
            y = self.predict_one(x)
            predicted.append(y)
        ys = np.array(predicted)
        return ys
    

# The Condensed Nearest Neighbor Algorithm (CNN)
# Chapter 6, problem 6.13
class CNN:
    def __init__(self, k):
        self.k = k # Number of nearest neighbors

    def init_cnn(self, X):
        # Initialize the condensed set S with self.k data 
        # points randomly selected from X

        # Return the indices for S
        N = X.shape[0]
        S_idx = np.random.choice(N, self.k)
        return S_idx

    def find_inconsistency(self, X, y, cnn, onn):
        #Is the condensed set training data consistent? 
        found = False        
        for ix, x1 in enumerate(X): # It can be a point in S as well
            x1 = x1.reshape(1, -1)
            y1 = cnn.predict_one(x1) # O(K)
            yo = onn.predict_one(x1)
            if y1 != yo:
                found = True
                #print('Found diff:', ix, x1, y1, yo)
                break
        inconsistent_idx = ix if found else None
        return inconsistent_idx, x1, yo

    def setup_cnn(self, X, y, S_idx):
        # Build a NearestNeighbors classifier based on 
        # the condensed nearest neighbors

        S = X[S_idx, :]
        ys = y[S_idx]
        cnn = NearestNeighbors(S, ys, self.k)
        return cnn

    def augment_S(self, X, y, inconsistent_y, neighbors_idx, S_idx):
        # The purpose is to find a point different from 
        # and nearest to inconsistent_idx

        #inconsistent_x = X[inconsistent_idx, :].reshape(-1, d)
        # inconsistent_y = y[inconsistent_idx] #This is wrong, should be the y prediced by onn
        #inconsistent_y = onn.predict_one(inconsistent_x)

        found = False
        for ix in neighbors_idx:
            if ix in S_idx: #Find x' not in S already
                continue
            if y[ix] == inconsistent_y: #Found the new point, should always find
                found = True 
                break
        if found:
            #print('Found a new idx: ', ix)
            S_idx = np.append(S_idx, ix)
        else:
            print("Can't find a new idx.")
        return S_idx

    def find_cnn(self, X, y):
        N, _ = X.shape
        S_idx = self.init_cnn(X)
        onn = NearestNeighbors(X, y, self.k)
        while True:
            old_s = len(S_idx)
            #print('Size of S_idx: ', old_s)
            cnn = self.setup_cnn(X, y, S_idx)
            inconsistent_idx, inconsistent_x, inconsistent_y = self.find_inconsistency(X, y, cnn, onn)
            #print('inconsistent idx: ', inconsistent_idx)
            if inconsistent_idx is None:
                break
            # Find the neighbors from nearest to farest
            neighbors_idx = onn.find_nn_idx(inconsistent_x, N)
            #print('Input inconsistency: ', inconsistent_idx, inconsistent_y)
            #print('NUmber of neighbors: ', len(neighbors_idx), neighbors_idx[:10])

            S_idx = self.augment_S(X, y, inconsistent_y, neighbors_idx, S_idx)
            if len(S_idx) == old_s:
                print('No new point added into S. Exit.')
                break
        #print('Final S_idx: ', S_idx)
        S = X[S_idx, :]
        Sy = y[S_idx]
        return S_idx, S, Sy




# Algorithm to construct condensed nearest neighbors using influence set
# as specified in Problem 6.12, use 1-NN here for simplicity
class InfluenceCNN():
    def __init__(self):
        pass
    
    def find_diff_cls_nn(self, D, y):
        """Find the nearest neighbor of each point from points
        with different class
        """
        N = len(y)
        min_ids = {}
        for i in np.arange(N):
            min_id = None
            min_D = math.inf
            for j in np.arange(N):
                if i == j:
                    continue
                if y[i] == y[j]:
                    continue
                if D[i, j] < min_D:
                    min_D = D[i,j]
                    min_id = j
            min_ids[i] = min_id
        return min_ids
                

    def construct_influence_sets(self, X, y, D, diff_cls_nn):
        S = {}
        N = len(y)
        for i in np.arange(N):
            Si = set()
            for j in np.arange(N):
                if i == j:
                    continue
                if y[i] != y[j]: #Need find points of the same class
                    continue
                min_diff_cls_D = diff_cls_nn[j]
                if D[i,j] < D[j, min_diff_cls_D]:
                    Si.add(j)
            if len(Si) > 0:
                S[i] = Si

        sorted_x = sorted(S.items(), key=lambda item: len(item[1]))
        ordered_S = collections.OrderedDict(sorted_x)
        return ordered_S

    def remove_largest(self, C_idx, S):
        # Remove the elements of the largest influence set from C
        C_idx = set(C_idx) - set(S)
        return C_idx

    def update_influence_sets(self, ordered_S, S1, id1):
        # Remove the elements of the S1 from other influence sets in S
        removal = set(S1)
        removal.add(id1)
        S = {}

        for ix, infS in ordered_S.items():
            if ix in S1: #Skip the points that have been removed
                continue
            k = set(infS) - removal
            #print('ix: ', ix, 'k ', k)
            if len(k) > 0:
                S[ix] = k
        sorted_x = sorted(S.items(), key=lambda item: len(item[1]))
        ordered_S = collections.OrderedDict(sorted_x)
        return ordered_S

    def find_cnn(self, X, y):

        D = build_distance_matrix(X)
        diff_cls_nn = self.find_diff_cls_nn(D, y)
        C_idx = np.arange(len(y)) #Initialize C with original data set
        ordered_S = self.construct_influence_sets(X, y, D, diff_cls_nn)

        while len(ordered_S) > 0:
            id1, S1 = ordered_S.popitem() #The largest influence set
            C_idx = self.remove_largest(C_idx, S1)
            ordered_S = self.update_influence_sets(ordered_S, S1, id1)
        

        C_idx = np.array(list(C_idx))

        C = X[C_idx, :]
        Cy = y[C_idx]
        return C_idx, C, Cy

def calc_distance_to_set(x, points):
    """Calculate the distance of a point 'x' to a set of points.
    it's defined as the distance to its nearest neighbor in the set
    """

    x = x.reshape(1, -1)
    #points = np.array(points).reshape(-1, 2)
    idx, _ = find_nn_idx(x, points, 1)
    nn = points[idx].reshape(1, -1)
    dis = distance.cdist(x, nn, 'euclidean')    
    return dis, idx

def create_separated_centers(X, M):
    """Create M separated centers for data points in X
    using a greedy approach as described in page 16 of 
    book: Learn From Data: A Short Course. Chapter 6.    
    """

    N, d = X.shape
    first_center = np.random.choice(N, 1)
    centers = np.zeros((M, d)) 
    centers[0] = X[first_center]
    for ic in range(1, M):
        dist_to_centers = -math.inf
        candidate_center = None
        #print('------- center: ', ic, centers[:ic], ' ------')
        for x in X:
            if x in centers:
                continue
            
            dist, _ = calc_distance_to_set(x, centers[:ic].reshape(-1, d))
            #print('x: ', x, ' dist: ', dist, '  dist_to_centers', dist_to_centers)
            if dist > dist_to_centers:
                dist_to_centers = dist
                candidate_center = x
        centers[ic] = candidate_center
        #print('---- Found center: ', candidate_center)
    return np.array(centers)

def update_to_voronoi_centers(X, centers):
    """Given data points in X with initial 'centers'.
    Update the centers with Voronoi definition and compute
    their radii
    """
    clusters = {}
    for ix, x in enumerate(X):
        closest_d = math.inf
        min_cx = None
        for cx, c in enumerate(centers):
            d = distance.cdist(x.reshape(1, -1), c.reshape(1, -1), 'euclidean')
            if d < closest_d:
                if clusters.get(cx, None) is None:
                   clusters[cx] = []
                min_cx = cx
                closest_d = d
        clusters[min_cx].append(ix)
    
    # Compute Voronoi center
    avgs = {}
    for c_id, points in clusters.items():
        #print('points: ', c_id, points)
        avg = np.mean(np.array(X[points]), axis=0)
        avgs[c_id] = avg

    avgs = np.array([c for _, c in avgs.items()])
    rs = {}
    for c_id, points in clusters.items():
        avg = avgs[c_id]
        rs[c_id] = np.max(np.linalg.norm(avg - np.array(X[points]), axis=1))
    return avgs, rs 

def get_current_clusters(X, centers):
    """Assign each point to a center
    """
    clusters = {}
    for ix, x in enumerate(X):
        closest_d = math.inf
        min_cx = None
        for cx, c in enumerate(centers):
            d = distance.cdist(x.reshape(1, -1), c.reshape(1, -1), 'euclidean')
            if d < closest_d:
                if clusters.get(cx, None) is None:
                   clusters[cx] = []
                min_cx = cx
                closest_d = d
        clusters[min_cx].append(ix)
            
    return clusters

def create_data_partitions(X, M, iters = 10, tol = 1.0e-3):
    """
    Given data points in X, create a partition of M clusters
    using a greedy approach as described in page 16 of 
    book: Learn From Data: A Short Course. Chapter 6.
    """

    centers = create_separated_centers(X, M)
    prev = centers
    for it in range(iters):
        centers, radii = update_to_voronoi_centers(X, centers)
        diff = np.linalg.norm(centers - prev)
        if diff <= tol:
            print(f'Found converge in partition at iteration: {it}')
            break
        prev = centers
    clusters = get_current_clusters(X, centers)
    return centers, radii, clusters


# Simple Branch and Bound Approach
def find_closest_center(query, centers, radii):
    min_d = math.inf
    min_ic = None
    distances = dist(query.reshape(-1, 2), centers).ravel()
    nn_ids = np.argsort(np.array(distances))
    for ic in nn_ids:
        if distances[ic] > radii[ic]:
            min_ic = ic
            min_d = distances[ic]
            break
    return min_ic, min_d, distances  

def find_nn_in_cluster(query, clus):
    nn_id, dp = find_nn_idx(query.reshape(-1, 2), clus, 1)
    nn_pt = clus[nn_id]
    return nn_pt, dp

def simple_brach_bound(query, X, clusters, centers, radii):
    center_id, _, dists = find_closest_center(query, centers, radii)

    count1 = 0
    count2 = 0

    if center_id is None:
        count1 += 1
        nn_id, dp = find_nn_idx(query.reshape(-1, 2), X, 1)        
        nn_pt = X[nn_id]
        return nn_pt, count1, count2
        
    nn_pt, dp = find_nn_in_cluster(query, X[clusters[center_id]])
    
    for ic, _ in enumerate(centers):
        if ic == center_id:
            continue
        dc = dists[ic]
        bound_criteria = dc - radii[ic]
        if dp <= bound_criteria: #Satisfy the bound, skip current cluster
            continue

        count2 += 1
        nn_pt2, dp2 = find_nn_in_cluster(query, X[clusters[ic]])
        if dp2 < dp:
            nn_pt = nn_pt2
            dp = dp2
    
    return nn_pt, count1, count2


def lloyd_kmeans(X, k, iters = 10, tol = 1.0e-3):
    """
    Given data points in X, create a partition of k clusters
    using Lloyd's Algorithm as described in page 32 of 
    book: Learn From Data: A Short Course. Chapter 6.
    """

    centers = create_separated_centers(X, k)
    prev_Ein = math.inf
    for it in range(iters):
        clusters = get_current_clusters(X, centers)
        centers_d = update_centroids(X, clusters)
        E_in = calc_in_sample_error(X, clusters, centers_d)
        #convert the dict 'centers' to array 
        for cid, center in centers_d.items():
            centers[cid] = center
        diff = np.abs(E_in - prev_Ein)
        prev_Ein = E_in
        if diff <= tol:
            #print(f'Found converge in partition at iteration: {it}')
            break
    return centers, clusters, prev_Ein

def update_centroids(X, clusters):
    """Given a set of clusters, compute their centroids
    """   

    centers = {}
    for c_id, points in clusters.items():
        avg = np.mean(np.array(X[points]), axis=0)
        centers[c_id] = avg

    #centers = np.array([c for _, c in centers.items()])
    return centers

def calc_in_sample_error(X, clusters, centers):
    Ein = 0
    _, d = X.shape
    for cid, points in clusters.items():
        #print(cid, type(points), points[0], d, type(X[points]))
        pts = np.array(X[points]).reshape(-1, d)
        c = centers[cid].reshape(-1, d)
        dist = np.linalg.norm(pts - c)
        Ein += dist**2
    return Ein