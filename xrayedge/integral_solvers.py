import numpy as np
from scipy import linalg
import os


def cheb_points(n, a=-1., b=1.):
    """Chebychev nodes on interval [a, b]"""
    assert(n > 1)
    assert(b > a)
    return (1. - np.cos((np.pi * np.arange(n)) / (n - 1))) * ((b - a) / 2.) + a

def test_cheb_points():
    np.testing.assert_allclose(cheb_points(3), [-1., 0., 1.], atol=1e-15)
    np.testing.assert_allclose(cheb_points(3, -2., 16.), [-2., 7., 16.], atol=1e-15)

### Quasi-Dyson equation solver

### loading Lagrange convolution integrals for the Chebychev method
def load_lagrange_convol_integrals():
    module_dir = os.path.dirname(__file__)

    grea = [None, None]
    less = [None, None]

    N = 2
    while True:
        filename = os.path.join(module_dir, f'data/lagrange_convol_integrals_N={N}.npz')
        try:
            with np.load(filename) as data:
                grea.append(data['grea'])
                less.append(data['less'])
        except FileNotFoundError:
            break

        N += 1
    
    if N == 2:
        raise FileNotFoundError("No Lagrange convol integrals file found.")

    return grea, less

lagrange_integrals_grea, lagrange_integrals_less = load_lagrange_convol_integrals()


def solve_pseudo_dyson(g_less, g_grea, t, V, N, method='cheb'):
    """
    Solve the following equation (for 0 <= u <= t):

    f(u) = g(u - t) - V \int_0^t dv g(u - v) f(v)

    with g(x) = \theta(x) g^>(x) + \theta(-x) g^<(x)

    f is solved on a Chebyshev grid of N points.
    g_less and g_grea should be vectorized.

    returns u_grid, f
    """
    assert(t > 0.)
    assert(N > 1)

    if method == 'cheb':
        if N >= len(lagrange_integrals_grea):
            raise RuntimeError(f"Lagrange convolution integrals have not been computed for N={N}")
        t_array = cheb_points(N, 0., t)

        mat_M = np.empty((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):

                mat_M[n, m] = np.dot(g_grea(t_array), lagrange_integrals_grea[N][m, n, :])
                mat_M[n, m] += np.dot(g_less(-t_array), lagrange_integrals_less[N][m, n, :])

            mat_M[:, m] *= V * t
            mat_M[m, m] += 1.

        vec_b = g_less(t_array - t)

    elif method == 'trapz':
        t_array, delta = np.linspace(0., t, N, retstep=True)
        gg = g_grea(t_array)
        gl = g_less(-t_array)

        mat_M = np.zeros((N, N), dtype=complex)

        for n in range(N):
            if n > 0:
                for m in range(n + 1):
                    if m < n:
                        mat_M[n, m] += gg[n - m] / 3. + gg[n - m - 1] / 6.

                    if m > 0:
                        mat_M[n, m] += gg[n - m] / 3. + gg[n - m + 1] / 6.

            if n < N - 1:
                for m in range(n, N):
                    if m < N - 1:
                        mat_M[n, m] += gl[m - n] / 3. + gl[m - n + 1] / 6.

                    if m > n:
                        mat_M[n, m] += gl[m - n] / 3. + gl[m - n - 1] / 6.

            mat_M[n, :] *= V * delta
            mat_M[n, n] += 1.

        vec_b = g_less(t_array - t)

    else:
        raise ValueError(f'Method "{method}" not found.')

    return t_array, linalg.solve(mat_M, vec_b)

def test_solve_pseudo_dyson_cheb():
    V = 2.
    t = 3.

    def cst_func(c):
        return np.vectorize(lambda x: c)

    time, phi = solve_pseudo_dyson(cst_func(1.), cst_func(1.), t, V, 10, method='cheb')
    np.testing.assert_allclose(phi, 1. / (1. + V * t))


    time, phi = solve_pseudo_dyson(cst_func(0.), cst_func(1.), t, V, 10, method='cheb')
    np.testing.assert_allclose(phi, 0.)


    time, phi = solve_pseudo_dyson(cst_func(1.), cst_func(0.), t, V, 20, method='cheb')
    np.testing.assert_allclose(phi, np.exp(V * (time - t)))


def test_solve_pseudo_dyson_trapz():
    V = 2.
    t = 3.

    def cst_func(c):
        return np.vectorize(lambda x: c)

    time, phi = solve_pseudo_dyson(cst_func(1.), cst_func(1.), t, V, 10, method='trapz')
    np.testing.assert_allclose(phi, 1. / (1. + V * t))


    time, phi = solve_pseudo_dyson(cst_func(0.), cst_func(1.), t, V, 10, method='trapz')
    np.testing.assert_allclose(phi, 0.)


    time, phi = solve_pseudo_dyson(cst_func(1.), cst_func(0.), t, V, 1000, method='trapz')
    np.testing.assert_allclose(phi, np.exp(V * (time - t)), rtol=1e-4, atol=1e-4)

### Cumulated adaptative integral



def cum_semiinf_adpat_simpson(f, scale=1., tol=1e-8, slopetol=1e-8, extend=True, maxfeval=100000):
    """
    Adaptative simpson cumulative integral (antiderivative) of `f`, starting from x=0 toward x > 0.
    Assumes `f` goes to a constant value at x=+infinity.

    If `extend` is True, samples increasing values of x until the slope of f is smaller than `slopetol`.
    Refine the sampling until the polynomial interpolation corresponding to the Simpson rule proves accurate up to `tol`.
    `scale` gives the size of the initial Simpson segment.

    Returns x_cumint, cumint, err
    `cumint` is an array containing the antiderivative at points `x_cumint`.
    `err` is an upper bound of the integration error.
    """
    bound_left = 0.

    a = bound_left
    b = bound_left + np.abs(scale)
    m = (a + b) / 2
    x = [a, m, b]
    y = [f(a), f(m), f(b)]

    i = 0
    feval = 3

    while i < len(x):

        if feval >= maxfeval:
            print("/!\ max number of function evaluations reached. Stopped iterating.")
            break

        if i == len(x) - 1: # end segment

            if extend: # check slope and add segment if necessary
                slope = max((y[-5] - 4 * y[-3] + 3 * y[-1]) / (x[-1] - x[-5]), (y[-3] - 4 * y[-2] + 3 * y[-1]) / (x[-1] - x[-3]))

                if np.abs(slope) < slopetol:
                    break # finished!

                else: # add segment
                    a = x[-1]
                    l = a - x[-5]
                    x.append(a + l)
                    y.append(f(x[-1]))
                    x.append(a + 2 * l)
                    y.append(f(x[-1]))
                    feval += 2

                    # then work on new segment

            else: # we don't check slope
                break # finished!

        # bulk segment
        a1_estimate = (3 * y[i] + 6 * y[i + 1] - y[i + 2]) / 8.
        a2_estimate = (3 * y[i + 2] + 6 * y[i + 1] - y[i]) / 8.

        a1 = (x[i] + x[i + 1]) / 2
        a2 = (x[i + 1] + x[i + 2]) / 2

        x.insert(i + 1, a1)
        y.insert(i + 1, f(a1))
        x.insert(i + 3, a2)
        y.insert(i + 3, f(a2))
        feval += 2

        err = max(np.abs(y[i + 1] - a1_estimate), np.abs(y[i + 3] - a2_estimate))

        if err < tol:
            i += 4 # go to next segment
        # else keep working on this segment

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=complex)
    # print("feval:", feval)

    # perform the cumulative integral
    cumint = (x[2::2] - x[:-1:2]) * (y[:-1:2] + 4 * y[1::2] + y[2::2]) / 6.
    cumint = np.append([0.], cumint)
    cumint = np.cumsum(cumint)
    x_cumint = x[::2]

    # upper bound error on integral
    cumint_2 = (x[4::4] - x[:-1:4]) * (y[:-1:4] + 4 * y[2::4] + y[4::4]) / 6.
    cumint_2 = np.cumsum(cumint_2)
    err = np.abs(cumint[2::2] - cumint_2)
    # x_err = x[4::4]

    return x_cumint, cumint, max(err)
        

def test_cum_semiinf_adpat_simpson():
    def f(x):
        return 10 * np.exp(-x * 3.) * np.sin(x) + (3 * x) / (2. * x + 6.)

    x_cum, cum, err = cum_semiinf_adpat_simpson(f, 20., tol=1e-10)
    
    ref = 21.834031327325317 # integral from 0 to 20

    i = np.argmin(np.abs(x_cum - 20.))
    assert(x_cum[i] == 20.)
    np.testing.assert_allclose(cum[i], ref, atol=1e-10)
    np.testing.assert_allclose(cum[i], ref, atol=err)


if __name__ == "__main__":
    print("Running tests...")
    test_cheb_points()
    test_solve_pseudo_dyson_cheb()
    test_solve_pseudo_dyson_trapz()
    test_cum_semiinf_adpat_simpson()
    print("Success.")