import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import functools
import h5py
from sklearn.model_selection import StratifiedShuffleSplit
import scipy
from scipy.linalg import sqrtm
from sklearn.utils.extmath import svd_flip


def generate_random_numbers01(N, dim, max_v = 10000):
    """
    Generate random numbers between 0 and 1
    max_v: maximum value used to generate random integers
    """
    random_ints = np.random.randint(max_v, size=(N, dim))
    init_lb = 0
    return (random_ints - init_lb)/(max_v - 1 - init_lb)

def generate_random_numbers(N, dim, max_v, lb, ub):
    """
    Generate random numbers between 'lb' and 'ub'
    """
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

    #X = X.reshape(-1, 1)  # Do I need to do this? 
    poly = PolynomialFeatures(q)
    res = poly.fit_transform(X)
    return res

def generate_target_coefficients(Qf, mu = 0, std = 1):
    # coefficients of the Legendre polynomials in the target function, problem 4.4
    mu, std = 0, 1
    aqs = np.random.normal(mu, std, Qf+1)
    normalized_aqs = normalize_legendre_coefficients(aqs)
    return tuple(normalized_aqs.flatten()) #make it hashable

def generate_data_set(N, aqs, sigma_square, tol = 0.0):
    # Generate random x samples for Problem 4.4.
    #
    #aqs: Coefficients for Legendre polynomials
    sigma = np.sqrt(sigma_square)
    max_v = 1000 # The range of integers used to generate random numbers
    dim = 1
    xs = generate_random_numbers(N, dim, max_v, -1 + tol, 1 - tol)
    epsilons = np.random.normal(0, 1, N).reshape(xs.shape)

    ys = calc_legendre_array(aqs, xs)
    ys = ys + sigma * epsilons
    #print('epsilon: ', epsilons.shape, 'ys: ', ys.shape, sigma, 'xs: ', xs.shape)
    return xs, ys

def calc_pred(w, test_xs, poly_transform=True):
    Z = test_xs
    if poly_transform:
        deg = w.shape[0] - 1
        Z = polynomial_transform(deg, test_xs)
    test_pred = np.matmul(Z, w)
    return test_pred

def calc_Eout(w, test_xs, test_ys, poly_transform=True):
    test_pred = calc_pred(w, test_xs, poly_transform)
    test_err = (test_pred - test_ys)
    E_out = np.matmul(test_err.transpose(), test_err).flatten()/test_xs.shape[0]
    return E_out


def rotate(X, theta, center = None):
    """Rotate the input point X of angle 'theta'
    N.B. Positive theta is considered as counter clockwise
    
    'center': This is the rotation center if specified
    """
    if not center:
        center = np.zeros(X.shape)
    r = np.linalg.norm(X - center)
    newX = np.array([r*np.cos(theta), r*np.sin(theta)])
    return center + newX

def generate_equal_spaced_points_on_circle(center, radius, N, starting_point = None):
    # LFD Problem 6.13
    # Generate N equally spaced points on the circule specified
    # by 'center' and 'radius'
    # If 'starting_point' is specified, use it as the first point

    x0, y0 = center
    if not starting_point:
        starting_point = np.array([x0 + radius, y0])
    
    # Angle for each arc
    points = []
    theta = 2*math.pi/N
    for idx in np.arange(N):
        theta1 = theta * idx
        p = rotate(starting_point, theta1)    
        points.append(p)
    return np.array(points)

def calc_cum_probs(probs):
    return np.cumsum(probs)

def sort_rnd_numbs_into_bins(rnd_numbers, probs):
    """
    Sort an array of random numbers between [0,1]
    into bins according to their probabilities
    """
    cum_probs = calc_cum_probs(probs)
 
    sels = {}
    for idx, cum_prob in enumerate(cum_probs):
        lb = 0 if idx == 0 else cum_probs[idx-1]
        ub = cum_prob
        sub = rnd_numbers[np.logical_and(rnd_numbers >= lb, 
                                         rnd_numbers <= ub)]
        sels[idx] = sub
    return sels

def generate_gmm(means, covs, probs, N):
    """Generate Gaussian Mixture Model Data
    Parameters
    ==========
    means: np array
        The means of Gaussian distributions
    covs: array of 2D matrices
        The covariance matrices of Gaussian distributions
    probs: np array
        The i-th element is the probability of picking the i-th Gaussian
    N: int
        The total number of points
    """
    # Number of Gaussians
    #N_gau = means.size
    # Generate random numbers between [0,1]
    gaussian_selection = generate_random_numbers01(N, 1)
    # Put the random numbers into bins according to their probabilities
    binned_nums = sort_rnd_numbs_into_bins(gaussian_selection, probs)
    # We only need the total count of numbers for each Gaussian
    binned_counts = {idx: len(nums) for idx, nums in binned_nums.items()}
    gaussians = {}
    for ix, count in binned_counts.items():
        mean = means[ix]
        cov = covs[ix]
        gs = np.random.multivariate_normal(mean, cov, count)
        gaussians[ix] = gs
    return gaussians


# USPS Zip Code Data for Handwritten Recognition
def load_zip_data(zip_data_path):
    """Load the USPS zip code data
    https://www.kaggle.com/bistaumanga/usps-dataset/data
    """
    with h5py.File(zip_data_path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    
    return X_tr, y_tr, X_te, y_te

def sample_zip_data(X, y, train_size, splits):
    sss = StratifiedShuffleSplit(n_splits=splits, train_size=train_size, random_state=0)
    sss.get_n_splits(X, y)

    data_indices = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    
        data_indices.append([X_train, y_train, X_test, y_test])
    return data_indices

# Deal with ZIP code data
def split_zip_data(zip_data_path, splits = 1, train_size = 500):
    # Split the raw data into train and test
    # splits: specify the number of random splits for each train-test pair
    X_tr, y_tr, X_te, y_te = load_zip_data(zip_data_path)
    train_size = train_size
    splits = splits
    data_splits = sample_zip_data(X_tr, y_tr, train_size, splits)
    return data_splits

def set_two_classes(y_train, y_test, digit):    
    # e.g. Classify digit '1' vs. not '1'
    y_train[y_train==digit] = 1
    y_test[y_test==digit] = 1
    
    y_train[y_train!=digit] = -1
    y_test[y_test!=digit] = -1
    return y_train, y_test

def calc_image_symmetry(X, img_w, img_h):
    """We define asymmetry as the average absolute difference between
    an image and its flipped versions, and symmetry as the negation of asymmetry

    X: Nxd: where N is the number of images, d is the number of pixels
    img_w, img_h: Image width and height, e.g. 16x16
    Then we have d = img_w x img_h
    """

    N, d = X.shape
    if d!= img_w*img_h:
        raise ValueError("Image width and height don't agree with data.")
    Xf = X.reshape(N, img_w, img_h)
    Xf = np.flip(Xf, axis=2)
    Xf = Xf.reshape(N, d)
    asy = np.abs(X - Xf)
    asy = np.mean(asy, axis = 1)
    sy = -asy
    return sy

def calc_image_intensity(X):
    """Compute the average intensity of an image
    X: Nxd: where N is the number of images, d is the number of pixels

    Return
    ret: Nx1 matrix
    """        
    
    ret = np.mean(X, axis=1)
    return ret


def compute_features(X_train, X_test):
    # Compute the symmetry and intensity for images
    img_w, img_h = 16, 16
    X_tr_sy = calc_image_symmetry(X_train, img_w, img_h)
    X_tr_int = calc_image_intensity(X_train)

    X_te_sy = calc_image_symmetry(X_test, img_w, img_h)
    X_te_int = calc_image_intensity(X_test)

    X_tr = np.hstack([X_tr_int.reshape(-1, 1), X_tr_sy.reshape(-1, 1)])
    X_te = np.hstack([X_te_int.reshape(-1, 1), X_te_sy.reshape(-1, 1)])
    return X_tr, X_te



# Input Centering
def input_centering(X):
    # Make the mean of X to be zero
    N, _ = X.shape
    mean_x = np.mean(X, axis = 0).reshape(1, -1)
    ones = np.ones((N,1))
    Z = X - np.matmul(ones, mean_x)
    return Z, mean_x

def input_whitening(X):
    # Center the data first
    N, _ = X.shape
    XX, _ = input_centering(X)
    COV = np.matmul(XX.transpose(), XX)/N
    sqrt_COV = sqrtm(COV)
    Z = np.matmul(XX, np.linalg.inv(sqrt_COV))
    return Z

class Whitening():
    def __init__(self, centering=True):
        self.centering = centering
        self.mean = None
        self.sqrt_COV = None
    
    def fit(self, X):
        N, _ = X.shape
        if self.centering:
            X, self.mean = input_centering(X)
        COV = np.matmul(X.transpose(), X)/N
        self.sqrt_COV = sqrtm(COV)
    
    def transform(self, X):
        if self.centering:
            X = X - self.mean
        Z = np.matmul(X, np.linalg.inv(self.sqrt_COV))
        return Z


class PCA:
    def __init__(self, top_k, centering=True):
        self.top_k = top_k
        self.centering = centering
        self.U, self.S, self.V, self.mean = None, None, None, None
        #PAC dimension reduction to top_k
        if top_k < 1:
            raise ValueError(f"The reduced dimension {top_k} has to be larger than 0")
        self.Vk = None

    def fit(self, X):
        if self.centering:
            XX, self.mean = input_centering(X)
        else:
            XX = X
        self.U, self.S, self.V = scipy.linalg.svd(XX, full_matrices=False)
        #Note, V is a Unitary matrix having right singular vectors as rows.
        self.V = self.V.transpose()
        self.Vk = self.V[:, :self.top_k]

    def transform(self, X):
        if self.centering:
            XX = X - self.mean
        else:
            XX = X
        Z = np.matmul(XX, self.Vk)
        return Z

    def reconstruct(self, X):
        if self.centering:
            XX = X - self.mean
        else:
            XX = X
        X_hat = np.matmul(XX, self.Vk)
        X_hat = np.matmul(X_hat, self.Vk.transpose())
        return X_hat

