import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import pandas as pd
import numpy as np
import math
from functools import partial

import cvxopt
import sklearn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge, Lasso

import data_util as du

def compute_H(X, lambda_t):
    # H(\lambda) = Z(Z^TZ+\lambda_t I )^{-1} Z^T
    xtx = np.matmul(X.transpose(), X) + lambda_t
    xtx_inv = np.linalg.inv(xtx)
    H = np.matmul(X, xtx_inv)
    H = np.matmul(H, X.transpose())
    return H

def compute_y_hat(H, y):
    y_hat = np.matmul(H, y)
    return y_hat

def calc_effective_dof(H):
    # Compute the effective degree of freedom in linear models
    # Page 33 in Chapter 9, example 9.5
    N = H.shape[0]
    d_eff = np.trace(H) - np.sum(H)/N
    return d_eff
    
def calc_error(w, xs, ys):
    c = 0
    for x, y in zip(xs, ys):
        prod = np.dot(w.T, x)*y
        if prod < 0:
            c +=1
    return c/len(ys)


def perceptron(points, dim, max_it=100, use_adaline=False, 
               eta = 1, randomize=False, print_out = True):
    w = np.zeros(dim+1)
    xs, ys = points[:,:dim+1], points[:,dim+1]
    num_points = points.shape[0]
    for it in range(max_it):
        correctly_predicted_ids=  set()
        idxs = np.arange(num_points)
        if randomize:
            idxs = np.random.choice(np.arange(num_points), num_points, replace=False)
        for idx in idxs:
            x, y = xs[idx], ys[idx]
            st = np.dot(w.T, x)
            prod = st*y #np.dot(w.T, x)*y
            if prod < -100: #avoid out of bound error
                st = -100
            threshold = 1 if use_adaline else 0
            st = st if use_adaline else 0
            if prod <= threshold:
                w = w + eta *(y-st)*x
                break #PLA picks one example at each iteration
            else:
                correctly_predicted_ids.add(idx)
        if len(correctly_predicted_ids) == num_points:
            break
    
    rou = math.inf
    R = 0
    c = 0
    for x, y in zip(xs, ys):
        prod = np.dot(w.T, x)*y
        if prod > 0:
            c +=1
        if prod < rou:
            rou = prod
        abs_x = np.linalg.norm(x)
        if abs_x > R:
            R = abs_x
    theoretical_t = (R**2) * (np.linalg.norm(w)**2)/rou/rou #LFD problem 1.3
    #w = w/w[-1]
    if print_out:
        print('Final correctness: ', c, '. Total iteration: ', it)
        print('Final w:', w)
    return w, it, theoretical_t

def pocket_algo(points, dim, max_it=100, eta = 1,
                randomized =False, print_out = True,
                test_points=None):
    w = np.zeros(dim+1)
    xs, ys = points[:,:dim+1], points[:,dim+1]
    if test_points is not None:
        test_xs, test_ys = test_points[:,:dim+1], test_points[:,dim+1]
    num_points = points.shape[0]
    sample_err = math.inf
    test_sample_err = math.inf
    w_ts, what_ts = np.zeros(max_it), np.zeros(max_it)
    test_w_ts, test_what_ts = np.zeros(max_it), np.zeros(max_it)
    wh = w #PLA w
    for it in range(max_it):
        idxs = np.arange(num_points)
        if randomized:
            idxs = np.random.choice(np.arange(num_points), num_points, replace=False)

        for idx in idxs:
            x, y = xs[idx], ys[idx]
            st = np.dot(wh.T, x)
            prod = st*y
            if prod <= 0:
                wh = wh + eta *y*x
                break

        in_sample_err = calc_error(wh, xs, ys)
        if test_points is not None:
            test_sample_err_it = calc_error(wh, test_xs, test_ys)
            test_w_ts[it] = test_sample_err_it

        w_ts[it] = in_sample_err
        
        if in_sample_err < sample_err:
            w = wh
            sample_err = in_sample_err
            what_ts[it] = sample_err
            if test_points is not None:
                test_sample_err = test_sample_err_it
            test_what_ts[it] = test_sample_err
        else:
            what_ts[it] = sample_err
            test_what_ts[it] = test_sample_err

    w = w/w[-1]
    if print_out:
        print('final Error Rate: ', sample_err)
        print('final normalized w:', w)
    return w, w_ts, what_ts, test_w_ts, test_what_ts, sample_err


def linear_regression(X, y):
    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT,X)), XT)
    w = np.matmul(x_pseudo_inv,y)
    return w


class LinearRegressionBase:
    def __init__(self, algo_name):
        self.algo_name = algo_name
        self.w = None

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def predict(self, X):
        pass
    
    def calc_error(self, X, y):
        pass

class LinearRegression(LinearRegressionBase):
    def __init__(self, algo_name, reg_type, reg_param, 
                poly_degree = None, solver=None, to_classify=False):
        super().__init__(algo_name)
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.poly_degree = poly_degree #If apply polynomial transformation first
        self.solver = solver
        self.to_classify = to_classify


    def fit(self, X, y):
        Z = X
        if self.poly_degree:
            Z = du.polynomial_transform(self.poly_degree, X)       

        if self.algo_name == 'lasso':
            self.w = lasso_fit(Z, y, self.reg_param, self.reg_type, self.solver)
        elif self.algo_name == 'ridge':
            self.w = ridge_fit(Z, y, self.reg_param, self.reg_type)
        elif self.algo_name == 'sklearn_lasso':
            self.w = sklearn_lasso(Z, y, self.reg_param)
        elif self.algo_name == 'sklearn_ridge':
            self.w = sklearn_ridge(Z, y, self.reg_param)
        elif self.algo_name == 'sklearn_lr':
            self.w = sklearn_linear_regression(Z, y)
        elif self.algo_name == 'pesudo_inv':
            self.w = pseudo_inv(Z, y, self.reg_param)
        else:
            raise ValueError("Not implemented")

    def predict(self, X):
        Z = X
        if self.poly_degree:
            Z = du.polynomial_transform(self.poly_degree, X)
        y_pred = np.matmul(Z, self.w)
        if self.to_classify:
            y_pred = np.sign(y_pred)
        return y_pred
    
    def calc_error(self, X, y):
        y_pred = self.predict(X)
        err = y_pred - y
        error = np.matmul(err.transpose(), err).flatten()/y.shape[0]
        return error

def pseudo_inv(X, y, reg):
    """Linear Regression Pseudo-Inverse Algorithm
    Learning from Data: A short course. Chapter 3, page 86

    """
    #N, _ = X.shape
    # Pre-append column of 1
    #c = np.ones((N, 1))
    #Xa = np.hstack((c, X))
    Xa = X
    t = np.matmul(Xa.transpose(), Xa)
    pseudo_inv = np.matmul(np.linalg.inv(t + reg), Xa.transpose())
    w = np.matmul(pseudo_inv, y.reshape((-1, 1)))
    return w.reshape(-1, 1)



def sklearn_ridge(X, y, lambda_t, fit_intercept = False, solver='auto'):
    #If added 1 into features, set fit_intercept = False
    #else, if you want to fit the intercept, set it to True
    # set solver = 'cholesky' to get analytical solution
    clf = Ridge(alpha=lambda_t, fit_intercept = fit_intercept, solver=solver)
    clf.fit(X, y)
    return clf.coef_.reshape(-1, 1)

def sklearn_lasso(X, y, lambda_t, fit_intercept = False):
    #If added 1 into features, set fit_intercept = False
    #else, if you want to fit the intercept, set it to True
    # This minimizes: \frac{1}{2N}|Xw-y|^2_2 + \lambda|w|_1
    clf = Lasso(alpha=lambda_t, fit_intercept = fit_intercept)
    clf.fit(X, y)
    return clf.coef_.reshape(-1, 1)    

def sklearn_linear_regression(X, y, fit_intercept = False):
    clf = sklearn.linear_model.LinearRegression(fit_intercept)
    clf.fit(X, y)
    return clf.coef_.reshape(-1, 1)

def lasso_fit_tikhonov(X, y, reg_param):
    raise ValueError("Not implemented")

def lasso_fit_ivanov(X, y, reg_param, solver=None):
    # Apply quadratic programming to solve this
    # solver is None or 'mosek'
    N = len(y) #number of samples
    d = X.shape[1]
    num_vars = 2*d
    ones = np.ones(d)
    zeros = np.zeros(d)
    id_zeros = np.zeros((d,d))
    XTX = np.matmul(X.transpose(), X) #(d+1)x(d+1)
    Pw = 2*XTX/N
    Pw = np.hstack([Pw, id_zeros]) 
    id2_zeros = np.hstack([id_zeros, id_zeros])
    P = np.vstack([Pw, id2_zeros])
    yTX = np.matmul(y.transpose(), X) #1x(d+1)
    qw = - 2*yTX.transpose()/N #(d+1)x1
    q = np.vstack([qw, zeros.reshape(-1, 1)])
    #number of variables in the quadratic optimization problem
    # We use auxiliary variables t_i with w_i, where |w_i| <= t_i
    # So \sum^{d+1}_{i=1} t_i \le reg_param (e.g. C)
    # w_i - t_i <= 0
    # -w_i - t_i <= 0
    # The variables are now [w_1,\dots,w_d, t_1,\dots,t_d]^T

    # constraint: \sum^{d+1}_{i=1} t_i \le reg_param
    sum_m = np.hstack([zeros, ones]).reshape(1, -1) #1x2d
    identity_m = np.identity(d)
    # contraints: w_i - t_i <= 0
    upper_m = np.hstack([identity_m, -identity_m]) #2dx2d
    # constraints: -w_i - t_i <= 0
    lower_m = np.hstack([-identity_m, -identity_m]) #2dx2d
    #print('sum_m: ', sum_m.shape, 'upper_m: ', upper_m.shape, 'lower_m: ', lower_m.shape)
    G = np.vstack([sum_m, upper_m, lower_m])

    h = np.zeros(num_vars + 1)
    h[0] = reg_param
    h = h.reshape(-1, 1)
    #print('G: ', G.shape, 'h: ', h.shape)
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    res = cvxopt.solvers.qp(P, q, G, h, solver = solver, options={'show_progress':False})
    if res['status'] != 'optimal':
        print("Couldn't find optimal solution with reg_param: ", reg_param)
        print('Final status: ', res['status'])
    w = res['x']
    #print('w: ', type(w), w)
    import struct
    w = np.array(w)
    w = w[:d,:]
    #print('w new: ', w.shape, w) 
    return w

def ridge_fit_tikhonov(X, y, reg_param):
    _, d = X.shape
    XTX = np.matmul(X.transpose(), X)
    #N.B. we are minimizing \frac{1}{N}|Xw-y|^2_2 + \frac{\lambda}{N}w^Tw
    id_m = np.identity(d)
    inv_X = np.linalg.inv(XTX + reg_param*id_m)
    invXXT = np.matmul(inv_X, X.transpose())
    w = np.matmul(invXXT, y)
    return w

"""
    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT,X)), XT)
    w = np.matmul(x_pseudo_inv,y)
"""

def calc_sum_of_squares(X, y, w):
    N = y.shape[0]
    XTX = np.matmul(X.transpose(), X)
    wTXTX = np.matmul(w.transpose(), XTX)
    wTXTXw = np.matmul(wTXTX, w)
    yTX = np.matmul(y.transpose(), X)
    yTXw = np.matmul(yTX, w)
    yTy = np.matmul(y.transpose(), y)
    ss = wTXTXw - 2 * yTXw + yTy
    return ss/N

def calc_ss_with_constraints(X, y, w, C):
    # Compute the sum of squares and quadratic constraint
    ss = calc_sum_of_squares(X, y, w).flatten()
    constraint = np.matmul(w.transpose(), w) - C
    res = np.array([ss, constraint.flatten()]).reshape(2, 1) #2x1
    return res

def calc_derivative_ss(X, y, w):
    N = y.shape[0]
    XTX = np.matmul(X.transpose(), X)
    XTXw = np.matmul(XTX, w)
    XTy = np.matmul(X.transpose(), y)
    res = XTXw - XTy 
    return res*2/N

def calc_derivative_ss_with_constraints(X, y, w):
    dss = calc_derivative_ss(X, y, w) #(num_features +1)x1
    d_constraint = 2*w # (num_features + 1)x1
    dss = dss.transpose()
    d_constraint = d_constraint.transpose()
    res = np.vstack([dss, d_constraint]) # 2x(num_features +1)
    return res

def calc_2nd_deriv_ss(X):
    N = X.shape[0]
    # row#1: df1/dx1, df1/dx2, df1/dx3, ...
    # row#2: df2/dx1, df2/dx2, df2/dx3, ...
    XTX = np.matmul(X.transpose(), X).transpose()
    res = 2*XTX/N #(num_features+1) x (num_features+1)
    return res

def calc_2nd_deriv_ss_with_constraints(X, z):
    deriv2_ss = z[0]*calc_2nd_deriv_ss(X)
    deriv2_constraint = z[1] * 2 * np.identity(X.shape[1])
    res = deriv2_ss + deriv2_constraint
    return res

def ridge_fit_ivanov(X, y, reg_param):
    # Apply convex programming to solve this problem
    _, num_features = X.shape
    def F(w = None, z = None):
        # z: 2x1
        # w: (num_features+1)x1
        if w is None:
            w0 = np.zeros((num_features, 1)) #any feasible point
            return 1, cvxopt.matrix(w0)
        w = np.array(w).reshape(num_features, 1)
        f = calc_ss_with_constraints(X, y, w, reg_param)        
        Df = calc_derivative_ss_with_constraints(X, y, w)
        f, Df = cvxopt.matrix(f), cvxopt.matrix(Df)
        if z is None:
            return f, Df
        H = calc_2nd_deriv_ss_with_constraints(X, z)
        H = cvxopt.matrix(H)
        return f, Df, H
    res = cvxopt.solvers.cp(F, options={'show_progress':False})
    if res['status'] != 'optimal':
        print("Couldn't find optimal solution with reg_param: ", reg_param)
        print('Final status: ', res['status'])
    w = res['x']
    w = np.array(w)
    return w

def lasso_fit(X, y, reg_param, reg_type, solver):
    w = None
    if reg_type == 'Tikhonov':
        w = lasso_fit_tikhonov(X, y, reg_param)
    elif reg_type == 'Ivanov':
        w = lasso_fit_ivanov(X, y, reg_param, solver)
    else:
        raise ValueError("Not implemented")
    return w
            
def ridge_fit(X, y, reg_param, reg_type):
    w = None
    if reg_type == 'Tikhonov':
        w = ridge_fit_tikhonov(X, y, reg_param)
    elif reg_type == 'Ivanov':
        w = ridge_fit_ivanov(X, y, reg_param)
    else:
        raise ValueError("Not implemented")
    return w            


class SVM(LinearRegressionBase):
    def __init__(self, is_soft=False):
        self.is_soft = is_soft #hard margin or soft margin
        self.w = None
        self.b = None

    def margin(self):
        w_norm = np.linalg.norm(self.w)
        return 1.0/w_norm

    def fit(self, X, y, solver=None):
        #This algo is implented according to the "Linear Hard-Margin SVM with QP"
        #on page 8-10 in Chapter 8 from "Learning From Data: A Short Course"
        
        # Apply quadratic programming to solve this
        # solver is None or 'mosek'

        #N.B. X doesn't have the 1 column here
        N, d = X.shape
        p = np.zeros((d+1, 1))
        c = np.ones((N, 1))
        dzeros = np.zeros((d,1))
        Qup = np.hstack((np.zeros((1,1)), dzeros.transpose() ))
        Qdown = np.hstack((dzeros, np.identity(d)))
        Q = np.vstack((Qup, Qdown))
        A = np.hstack((c, X))
        A = np.multiply(A, y.reshape(-1, 1))

        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(p)
        G = cvxopt.matrix(-A)
        h = cvxopt.matrix(-c)
        res = cvxopt.solvers.qp(P, q, G, h, solver = solver, options={'show_progress':False})
        if res['status'] != 'optimal':
            print(f"Couldn't find optimal solution!")
            print(f"Final status: {res['status']}")

        u = np.array(res['x'])
        self.b = u[0, :]
        self.w = u[1:, :]

    def predict(self, X):
        N, _ = X.shape
        c = np.ones((N, 1))
        A = np.hstack((c, X))
        u = np.vstack((self.b, self.w))
        pred = np.sign(np.matmul(A, u))
        return pred

    def calc_error(self, X, y):
        N, _ = X.shape
        c = np.ones((N, 1))
        XX = np.hstack((c, X))        
        u = np.vstack((self.b, self.w))
        error = calc_error(u, XX, y)
        return error


