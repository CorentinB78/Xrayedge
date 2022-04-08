#!python3
import numpy as np
from scipy import integrate


def cheb_points(n, a=-1.0, b=1.0):
    """
    Chebyshev points on interval [a, b]
    """
    assert n > 1
    assert b > a
    return (1.0 - np.cos((np.pi * np.arange(n)) / (n - 1))) * ((b - a) / 2.0) + a


np.testing.assert_allclose(cheb_points(3), [-1.0, 0.0, 1.0], atol=1e-15)
np.testing.assert_allclose(cheb_points(3, -2.0, 16.0), [-2.0, 7.0, 16.0], atol=1e-15)


def lagrange_convol_integral(N, m, n, k):
    """
    Computes convolution integrals between Lagrange polynomials on a Chebyshev grid.

    Given the N Chebyshev nodes x_1, ... x_N on the interval [0, 1],
    we call l_k the kth lagrange polynomial.
    This function computes the following integrals:

    Greater integrals: \int_0^{x_n} dx l_k(x_n - x) l_m(x)

    Lesser integrals: \int_{x_n}^1 dx l_k(x - x_n) l_m(x)
    """
    pts = cheb_points(N, 0.0, 1.0)
    xn = pts[n]

    def lagrange(j):
        xj = pts[j]

        def f(x):
            out = 1.0
            for k, xk in enumerate(pts):
                if k != j:
                    out *= (x - xk) / (xj - xk)
            return out

        return f

    lk = lagrange(k)
    lm = lagrange(m)

    def f_grea(x):
        return lk(xn - x) * lm(x)

    res_grea = integrate.quad(f_grea, 0, xn)

    def f_less(x):
        return lk(x - xn) * lm(x)

    res_less = integrate.quad(f_less, xn, 1.0)

    return res_grea[0], res_less[0]


def compute_all(N):

    grea = np.empty((N, N, N))
    less = np.empty((N, N, N))

    for m in range(N):
        for n in range(N):
            for k in range(N):
                g, l = lagrange_convol_integral(N, m, n, k)
                grea[m, n, k] = g
                less[m, n, k] = l

    return grea, less


if __name__ == "__main__":
    from os import path, remove
    import argparse

    def filename(N):
        return f"lagrange_convol_integrals_N={N}.npz"

    parser = argparse.ArgumentParser(
        description=f'Compute convolution integrals between Lagrange polynomials on a Chebyshev grid. It is useful for convoluting functions quickly. Data is stored in files "{filename("*")}".'
    )
    parser.add_argument(
        "Nmax",
        type=int,
        nargs=1,
        help="Max number N of Chebyshev points. The integrals are computed for all N <= Nmax",
    )
    parser.add_argument(
        "--time", action="store_true", help="Also plot execution time as function of N"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of all N. Existing files will be deleted.",
    )
    args = parser.parse_args()

    Nmax = args.Nmax[0]

    if args.time:
        from time import time
        from matplotlib import pyplot as plt

        times = []

    if Nmax < 2:
        print("Nothing to do.")
        exit

    for N in range(2, Nmax + 1):

        if path.exists(filename(N)):
            if args.force:
                remove(filename(N))
            else:
                print(f"N={N} already computed.")
                if args.time:
                    times.append(0.0)
                continue

        print(f"Computing N={N}...")

        if args.time:
            start = time()

        grea, less = compute_all(N)

        if args.time:
            times.append(time() - start)

        np.savez(filename(N), less=less, grea=grea)

    print("Done.")

    if args.time:
        print("times:", times)

        n = np.arange(2, len(times) + 2)
        plt.plot(n, times, "o-", label="data")
        plt.plot(n, 1e-4 * n**3, "--", label="N^3")
        plt.plot(n, 1e-5 * n**5, "--", label="N^5")

        plt.loglog()
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("time (arb. unit)")

        plt.show()
