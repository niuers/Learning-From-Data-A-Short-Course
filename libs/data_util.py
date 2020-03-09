import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import functools

def generate_random_numbers01(N, dim, max_v):
    """
    max_v: maximum value used to generate random integers
    """
    random_ints = np.random.randint(max_v, size=(N, dim))
    init_lb = 0
    return (random_ints - init_lb)/(max_v - 1 - init_lb)

def generate_random_numbers(N, dim, max_v, lb, ub):
    zero_to_one_points = generate_random_numbers01(N, dim, max_v)
    res = lb + (ub - lb)*zero_to_one_points
    return res

def generate_two_classes(N, dim, true_func, rn_func):
    cls1, cls2 = [], []
    while True:
        rn = rn_func(1, dim).flatten()
        if true_func(rn) > 0 and len(cls1) < N:
            cls1.append(rn)
        elif true_func(rn) < 0 and len(cls2) < N:
            cls2.append(rn)
        if len(cls1) == N and len(cls2) == N:
            break
    return np.asarray(cls1), np.asarray(cls2)   

import math


def generate_random_circle(N, r, max_v):
    """Generate random numbers in a circle, with radius r
    """
    rand_radius = generate_random_numbers(N, 1, max_v, 0, r)
    rand_degree = generate_random_numbers(N, 1, max_v, 0, 2.0*math.pi)
    return rand_radius, rand_degree

def generate_random_ring(N, r1, r2, max_v):
    """Generate random numbers in a ring between r1 and r2
    """
    radiuses = generate_random_numbers(N, 1, max_v, r1, r2)
    radians = generate_random_numbers(N, 1, max_v, 0, 2.0*math.pi)
    return radiuses, radians

def move_bottom_ring_and_assign(radiuses, radians, diffx, diffy):
    """
    Give the points within a ring, move the bottom half 'diffx' and 'diffy' along
    x and y directions respectively. Assign the bottom points to have sign -1
    """
    xs = radiuses * np.cos(radians)
    ys = radiuses * np.sin(radians)
    signs = np.ones(len(xs))

    for idx, r in enumerate(radiuses):
        rad = radians[idx]
        xi, yi = xs[idx], ys[idx]
        if rad > math.pi and rad < 2*math.pi:
            xs[idx] = xi + diffx
            ys[idx] = yi +  diffy
            signs[idx] = -1
    return xs, ys, signs


# This function call, legendre(k,x), if not careful, will take very long time for 
#large k. This requires k, x to be hashable, however np.array is not hashable.
@functools.lru_cache(128)
def legendre(k, x):
    """Calculate the Legendre polynomial of degree k at point x
    """
    if k == 0:
        return 1 #np.ones(x.shape) # x might be an array
    if k == 1:
        return x
    
    ret = (2*k-1)*x*legendre(k-1, x)/k - (k-1)*legendre(k-2, x)/k
    return ret

def normalize_legendre_coefficients(aqs):
    denominator = 0
    for q, _ in enumerate(aqs):
        denominator += 1.0/(2*q + 1)
    scale = np.sqrt(1/denominator)

    res = scale * aqs
    return res

@functools.lru_cache(128)
def legendre_poly(aqs, x):
    """Calculate the value of a polynomial (which is a sum of Legendre polynomials)
     at point x

    aqs: coefficients for the Legendre polynomials, a_0, a_1, ..., a_Q
    The degree of the final polynomial is: len(aqs) - 1
    """

    res = 0 #np.zeros(x.shape)
    #print('x.shape, aqs.shape', x.shape, aqs.shape)
    for k, aq in enumerate(aqs):
        #print('k: ', k)
        #lg = legendre(k, x)
        #print('lg.shape', lg.shape, 'res.shape: ', res.shape, 'aq: ', aq)
        #res = res + aq * lg.reshape(x.shape)
        #print('res: ', res.shape)
        res += aq * legendre(k, x)

    return res

def calc_legendre_array(aqs, xs):
    ys = np.zeros(xs.shape)
    for (i, j), x in np.ndenumerate(xs):
        ys[i,j] = legendre_poly(aqs, x)
    return ys

def polynomial_transform(q, X):
    """Transform the X using degree-q polynomials
    Return: A (N x (q+1)) matrix, where N = len(X)
    """

    X = X.reshape(-1, 1)
    poly = PolynomialFeatures(q)
    res = poly.fit_transform(X)
    return res

def generate_target_coefficients(Qf, mu = 0, std = 1):
    # coefficients of the Legendre polynomials in the target function
    mu, std = 0, 1
    aqs = np.random.normal(mu, std, Qf+1)
    normalized_aqs = normalize_legendre_coefficients(aqs)
    return tuple(normalized_aqs.flatten()) #make it hashable

def generate_data_set(N, aqs, sigma_square):
    # Generate random x samples
    sigma = np.sqrt(sigma_square)
    max_v = 1000 # The range of integers used to generate random numbers
    dim = 1
    xs = generate_random_numbers(N, dim, max_v, -1, 1)
    epsilons = np.random.normal(0, 1, N).reshape(xs.shape)

    ys = calc_legendre_array(aqs, xs)
    ys = ys + sigma * epsilons
    #print('epsilon: ', epsilons.shape, 'ys: ', ys.shape, sigma, 'xs: ', xs.shape)
    return xs, ys

def calc_pred(w, test_xs):
    deg = w.shape[0] - 1
    Z = polynomial_transform(deg, test_xs)
    test_pred = np.matmul(Z, w)
    return test_pred

def calc_Eout(w, test_xs, test_ys):
    test_pred = calc_pred(w, test_xs)
    test_err = (test_pred - test_ys)
    E_out = np.matmul(test_err.transpose(), test_err).flatten()/test_xs.shape[0]
    return E_out



    