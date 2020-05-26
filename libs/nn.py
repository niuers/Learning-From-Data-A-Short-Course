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

class NearestNeighbors:
    def __init__(self, X, y, k, problem_type='classification'):
        #X: Nxd matrix, where each row corresponds to a data point x in R^d
        self.X = X 
        self.y = y 
        self.k = k #number of nearest neighbors
        self.problem_type = problem_type

    def find_nn_idx(self, x):
        # Find the indexes of nearest neighbors for x
        distances = dist(x, self.X).ravel()
        order = np.argsort(np.array(distances))
        return order[:self.k]

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

    def find_inconsistency(self, X, y, cnn):
        #Is the condensed set training data consistent? 
        found = False
        
        for ix, x1 in enumerate(X): # It can be a point in S as well
            x1 = x1.reshape(1, -1)
            y1 = cnn.predict_one(x1) # O(K)
            if y1 != y[ix]:
                found = True
                break
        inconsistent_idx = ix if found else None
        return inconsistent_idx

    def setup_cnn(self, X, y, S_idx):
        # Build a NearestNeighbors classifier based on 
        # the condensed nearest neighbors

        S = X[S_idx, :]
        ys = y[S_idx]
        cnn = NearestNeighbors(S, ys, self.k)
        return cnn

    def augment_S(self, X, y, inconsistent_idx, S_idx):
        N, d = X.shape
        # The purpose is to find a point different from 
        # and nearest to inconsistent_idx
        nn = NearestNeighbors(X, y, N)
        inconsistent_x = X[inconsistent_idx, :].reshape(-1, d)
        inconsistent_y = y[inconsistent_idx]
        # Find the neighbors from nearest to farest
        neighbors_idx = nn.find_nn_idx(inconsistent_x)
        found = False
        for ix in neighbors_idx:
            if ix in S_idx: #Find x' not in S already
                continue
            if y[ix] == inconsistent_y: #Found the new point, should always find
                found = True 
                break
        if found:
            S_idx = np.append(S_idx, ix)
        return S_idx

    def find_cnn(self, X, y):
        
        S_idx = self.init_cnn(X)        
        while True:
            cnn = self.setup_cnn(X, y, S_idx)
            inconsistent_idx = self.find_inconsistency(X, y, cnn)
            if inconsistent_idx is None:
                break
            S_idx = self.augment_S(X, y, inconsistent_idx, S_idx)
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

