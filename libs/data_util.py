import numpy as np


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


def generate_numbers_from_normal(N, dim, mean, std):
    pass

def legendre_poly(x, q):
    """Calculate the Legendre polynomial of degree q at point x
    """
    pass

def polynomial(x, Q):
    """Calculate the value of a polynomial of degree Q at point x
    """
    pass

def generate_validation_experiment_dataset(Qf, N, sigma):
    target_fun = generate_target(Qf)
    xs = generate_xs(N)
    ys = generate_ys(xs, sigma)

    z2_s = transform(xs, 2)
    z10_s = transform(xs, 10)
    w2 = least_square_solution(z2_s, ys)
    w10 = least_square_solution(z10_s, ys)
    Eout_g2 = compute_eout(w2)
    Eout_g10 = compute_eout(w10)
    return Eout_g2, Eout_g10

    