import numpy as np
from scipy import linalg, interpolate
import os


def cheb_points(n, a=-1.0, b=1.0):
    """Chebychev nodes on interval [a, b]"""
    assert n > 1
    assert b > a
    return (1.0 - np.cos((np.pi * np.arange(n)) / (n - 1))) * ((b - a) / 2.0) + a


### Quasi-Dyson equation solver

### loading Lagrange convolution integrals for the Chebychev method
def load_lagrange_convol_integrals():
    module_dir = os.path.dirname(__file__)

    grea = [None, None]
    less = [None, None]

    N = 2
    while True:
        filename = os.path.join(module_dir, f"data/lagrange_convol_integrals_N={N}.npz")
        try:
            with np.load(filename) as data:
                grea.append(data["grea"])
                less.append(data["less"])
        except FileNotFoundError:
            break

        N += 1

    if N == 2:
        raise FileNotFoundError("No Lagrange convol integrals file found.")

    return grea, less


lagrange_integrals_grea, lagrange_integrals_less = load_lagrange_convol_integrals()

# TODO: check minus sign in g^<(-u_k)


def solve_pseudo_dyson(g_less, g_grea, t, V, N, method="cheb"):
    """
    Solve the following equation (for 0 <= u <= t):

    f(u) = g(u - t) - V \int_0^t dv g(u - v) f(v)

    with g(x) = \theta(x) g^>(x) + \theta(-x) g^<(x)

    f is solved on a Chebyshev grid of N points.
    g_less and g_grea should be vectorized.

    returns u_grid, f
    """
    assert t > 0.0
    assert N > 1

    if method == "cheb":
        if N >= len(lagrange_integrals_grea):
            raise RuntimeError(
                f"Lagrange convolution integrals have not been computed for N={N}"
            )
        t_array = cheb_points(N, 0.0, t)

        mat_M = np.empty((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):

                mat_M[n, m] = np.dot(
                    g_grea(t_array), lagrange_integrals_grea[N][m, n, :]
                )
                mat_M[n, m] += np.dot(
                    g_less(-t_array), lagrange_integrals_less[N][m, n, :]
                )

            mat_M[:, m] *= V * t
            mat_M[m, m] += 1.0

        vec_b = g_less(t_array - t)

    elif method == "multicheb":
        K = N
        M = (N - 1) // (K - 1)
        N = M * (K - 1) + 1
        print(M, N)
        print()

        delta = t / float(M)
        pts = cheb_points(K, 0.0, delta)

        t_array = np.empty(N, dtype=float)
        for i in range(M):
            t_array[i : i + K] = i * delta + pts

        mat_M = np.zeros((N, N), dtype=complex)
        for j in range(M):
            for jj in range(K):
                m = (K - 1) * j + jj
                # print(m)
                for i in range(M):
                    for ii in range(K):
                        n = ii + (K - 1) * i
                        print()
                        for l in range(M):
                            for ll in range(K):
                                if i - 2 <= l + j <= i:
                                    print(ll + (K - 1) * l)
                                    u_k = t_array[ll + (K - 1) * l]
                                    mat_M[n, m] += (
                                        g_grea(u_k)
                                        * lagrange_integrals_grea[K][jj, ii, ll]
                                    )
                                    mat_M[n, m] += (
                                        g_less(-u_k)
                                        * lagrange_integrals_less[K][jj, ii, ll]
                                    )

                mat_M[:, m] *= V * t
                mat_M[m, m] += 1.0

        vec_b = g_less(t_array - t)

    elif method == "trapz":
        t_array, delta = np.linspace(0.0, t, N, retstep=True)
        gg = g_grea(t_array)
        gl = g_less(-t_array)

        mat_M = np.zeros((N, N), dtype=complex)

        for n in range(N):
            if n > 0:
                for m in range(n + 1):
                    if m < n:
                        mat_M[n, m] += gg[n - m] / 3.0 + gg[n - m - 1] / 6.0

                    if m > 0:
                        mat_M[n, m] += gg[n - m] / 3.0 + gg[n - m + 1] / 6.0

            if n < N - 1:
                for m in range(n, N):
                    if m < N - 1:
                        mat_M[n, m] += gl[m - n] / 3.0 + gl[m - n + 1] / 6.0

                    if m > n:
                        mat_M[n, m] += gl[m - n] / 3.0 + gl[m - n - 1] / 6.0

            mat_M[n, :] *= V * delta
            mat_M[n, n] += 1.0

        vec_b = g_less(t_array - t)

    else:
        raise ValueError(f'Method "{method}" not found.')

    return t_array, linalg.solve(mat_M, vec_b)


### Cumulated adaptative integral


def cum_semiinf_adpat_simpson(
    f, scale=1.0, tol=1e-8, slopetol=1e-8, extend=False, maxfeval=100000
):
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
    bound_left = 0.0

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

        if i == len(x) - 1:  # end segment

            if extend:  # check slope and add segment if necessary
                slope = max(
                    (y[-5] - 4 * y[-3] + 3 * y[-1]) / (x[-1] - x[-5]),
                    (y[-3] - 4 * y[-2] + 3 * y[-1]) / (x[-1] - x[-3]),
                )

                if np.abs(slope) < slopetol:
                    break  # finished!

                else:  # add segment
                    a = x[-1]
                    l = a - x[-5]
                    x.append(a + l)
                    y.append(f(x[-1]))
                    x.append(a + 2 * l)
                    y.append(f(x[-1]))
                    feval += 2

                    # then work on new segment

            else:  # we don't check slope
                break  # finished!

        # bulk segment
        int1_estimate = (x[i + 2] - x[i]) * (5 * y[i] + 8 * y[i + 1] - y[i + 2]) / 24.0
        int2_estimate = (x[i + 2] - x[i]) * (5 * y[i + 2] + 8 * y[i + 1] - y[i]) / 24.0

        a1 = (x[i] + x[i + 1]) / 2
        a2 = (x[i + 1] + x[i + 2]) / 2

        x.insert(i + 1, a1)
        y.insert(i + 1, f(a1))
        x.insert(i + 3, a2)
        y.insert(i + 3, f(a2))
        feval += 2

        int1 = (x[i + 2] - x[i]) * (y[i] + 4 * y[i + 1] + y[i + 2]) / 6.0
        int2 = (x[i + 4] - x[i + 2]) * (y[i + 2] + 4 * y[i + 3] + y[i + 4]) / 6.0

        err = max(np.abs(int1 - int1_estimate), np.abs(int2 - int2_estimate))

        if err < tol:
            i += 4  # go to next segment
        # else keep working on this segment

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=complex)
    # print("feval:", feval)

    # perform the cumulative integral
    cumint = (x[2::2] - x[:-1:2]) * (y[:-1:2] + 4 * y[1::2] + y[2::2]) / 6.0
    cumint = np.append([0.0], cumint)
    cumint = np.cumsum(cumint)
    x_cumint = x[::2]

    # upper bound error on integral
    cumint_2 = (x[4::4] - x[:-1:4]) * (y[:-1:4] + 4 * y[2::4] + y[4::4]) / 6.0
    cumint_2 = np.cumsum(cumint_2)
    err = np.abs(cumint[2::2] - cumint_2)
    # x_err = x[4::4]

    return x_cumint, cumint, max(err)
