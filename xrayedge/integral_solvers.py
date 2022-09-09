import numpy as np
from scipy import linalg
from scipy.sparse.linalg import gmres, LinearOperator
from scipy import interpolate
import os


class QuasiToeplitzMatrix(LinearOperator):
    def __init__(self, c, r, corrections, dtype=complex):
        """
        Matrix of type Toeplitz + corrections on the first and last columns.

        Allows fast matrix vector product with FFT.

        Arguments:
            c -- first column for Toeplitz part
            r -- first row for Toeplitz part (r[0] is ignored)
            corrections -- pair of columns which are added as corrections
            dtype -- data type (default is complex)
        """
        self.c_r = (np.asarray(c), np.asarray(r))
        self.corrections = np.asarray(corrections)
        super().__init__(dtype, shape=(len(c), len(r)))

    def _matvec(self, x):
        out = linalg.matmul_toeplitz(self.c_r, x, check_finite=False)
        out[:] += x[0] * self.corrections[0] + x[-1] * self.corrections[1]
        return out

    def to_array(self):
        c, r = self.c_r
        out = linalg.toeplitz(c, r)
        out[:, 0] += self.corrections[0]
        out[:, -1] += self.corrections[1]
        return out


class BlockLinearOperator(LinearOperator):
    def __init__(self, linop_array, dtype=complex, check_shapes=False):
        self.blocks = np.asarray(linop_array, dtype=LinearOperator)

        assert self.blocks.ndim == 2
        supershape = self.blocks.shape

        shape = self.blocks[0, 0].shape
        if check_shapes:
            for op in self.blocks.flatten():
                assert op.shape == shape
        self.subshape = shape

        super().__init__(
            dtype, shape=(supershape[0] * shape[0], supershape[1] * shape[1])
        )

    def _matvec(self, x):
        out = np.zeros(self.shape[0], dtype=self.dtype)
        N, M = self.subshape
        for i in range(self.blocks.shape[0]):
            for j in range(self.blocks.shape[1]):
                out[i * N : (i + 1) * N] += self.blocks[i, j] @ x[j * M : (j + 1) * M]

        return out

    def to_array(self):
        out = np.empty(self.shape, dtype=self.dtype)
        N, M = self.subshape

        for i in range(self.blocks.shape[0]):
            for j in range(self.blocks.shape[1]):
                out[i * N : (i + 1) * N, j * M : (j + 1) * M] = self.blocks[
                    i, j
                ].to_array()

        return out


def cheb_points(n, a=-1.0, b=1.0):
    """Chebychev nodes on interval [a, b]"""
    assert n > 1
    assert b > a
    return (1.0 - np.cos((np.pi * np.arange(n)) / (n - 1))) * ((b - a) / 2.0) + a


class cpx_interp1d:
    """
    scipy.interpolate.interp1d for complex numbers.
    """

    def __init__(self, x, y, **kwargs) -> None:
        """
        scipy.interpolate.interp1d for complex numbers.
        """
        self.real_interp1d = interpolate.interp1d(x, np.real(y), **kwargs)
        self.imag_interp1d = interpolate.interp1d(x, np.imag(y), **kwargs)

    def __call__(self, x):
        return self.real_interp1d(x) + 1j * self.imag_interp1d(x)


### Quasi-Dyson equation solver

### loading Lagrange convolution integrals for the Chebychev method
def load_lagrange_convol_integrals():
    """
    Load the convolution integrals in data/ for the Chebychev method.
    """
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


def solve_quasi_dyson(
    g_less,
    g_grea,
    t,
    orb,
    orbitals,
    couplings,
    N,
    guess=None,
    method="trapz",
    tol_gmres=1e-10,
    atol_gmres=1e-10,
):
    """
    Solve the following equation (for 0 <= u <= t):

    f(u) = g(u - t) - V \int_0^t dv g(u - v) f(v)

    with g(x) = \theta(x) g^>(x) + \theta(-x) g^<(x)

    Arguments:
        g_less -- vectorized callable for g^<
        g_grea -- vectorized callable for g^>
        t -- positive float
        orb -- int, orbital index
        orbitals -- 1D array, list of orbital indices
        couplings -- 1D array, list of corresponding couplings
        V -- real or complex parameter
        N -- number of grid points on which f is computed

    Keyword Arguments:
        guess -- callable t -> 1D array. Only for GMRES method.
        method -- one of "cheb", "trapz", "trapz-LU", "trapz-GMRES" (default: {"trapz"})

    Returns:
        grid_pts -- 1D array, time coordinates
        f_values -- 2D array, values for f, shape (len(orbitals), N)
    """
    assert t > 0.0
    assert N > 1
    assert len(orbitals) == len(couplings)
    assert orb in orbitals
    orbitals = np.asarray(orbitals)
    couplings = np.asarray(couplings)
    nr_orb = len(orbitals)

    if method == "trapz":
        if N * nr_orb < 200:
            method = "trapz-LU"
        else:
            method = "trapz-GMRES"

    if method == "cheb":
        if orbitals != [orb]:
            raise ValueError(
                "multi-orbital coupling not implemented with 'cheb' method"
            )

        V = couplings[0]

        if N >= len(lagrange_integrals_grea):
            raise RuntimeError(
                f"Lagrange convolution integrals have not been computed for N={N}"
            )
        t_array = cheb_points(N, 0.0, t)
        gg = g_grea(orb, orb)(t_array) * V * t
        gl = g_less(orb, orb)(-t_array) * V * t

        mat_M = np.empty((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):

                mat_M[n, m] = np.dot(gg, lagrange_integrals_grea[N][m, n, :])
                mat_M[n, m] += np.dot(gl, lagrange_integrals_less[N][m, n, :])

            mat_M[m, m] += 1.0

        vec_b = g_less(orb, orb)(t_array - t)
        return t_array, linalg.solve(mat_M, vec_b)[None, :]

    elif method.startswith("trapz"):
        t_array, delta = np.linspace(0.0, t, N, retstep=True)

        blocks = np.empty((nr_orb, nr_orb), dtype=QuasiToeplitzMatrix)

        for j, orb_a in enumerate(orbitals):
            for k, orb_b in enumerate(orbitals):

                gg = g_grea(orb_a, orb_b)(t_array) * couplings[k] * delta / 6.0
                gl = g_less(orb_a, orb_b)(-t_array) * couplings[k] * delta / 6.0

                r = np.empty(N, dtype=complex)
                c = np.empty(N, dtype=complex)
                r[1:N] = gl[1:N] * 4.0 + gl[0 : N - 1]
                r[1 : N - 1] += gl[2:N]
                c[1:N] = gg[1:N] * 4.0 + gg[0 : N - 1]
                c[1 : N - 1] += gg[2:N]
                c[0] = (gg[0] + gl[0]) * 2.0 + (gg[1] + gl[1])

                # boundary corrections
                correc_0 = gg[0:N] * 2.0
                correc_0[0 : N - 1] += gg[1:N]

                correc_1 = gl[N - 1 :: -1] * 2.0
                correc_1[1:N] += gl[N - 1 : 0 : -1]

                if j == k:
                    c[0] += 1.0

                blocks[j, k] = QuasiToeplitzMatrix(c, r, (-correc_0, -correc_1))

        mat_M = BlockLinearOperator(blocks, check_shapes=True)

        vec_b = np.empty(N * nr_orb, dtype=complex)
        for j, orb_a in enumerate(orbitals):
            vec_b[j * N : (j + 1) * N] = g_less(orb_a, orb)(t_array - t)

        if method == "trapz-LU":
            mat_M = mat_M.to_array()
            return t_array, linalg.solve(mat_M, vec_b).reshape((nr_orb, N))

        elif method == "trapz-GMRES":

            guess_spl = None
            if guess is not None:
                guess_spl = np.empty_like(vec_b)
                for j in range(len(orbitals)):
                    guess_spl[j * N : (j + 1) * N] = guess(t_array)[j, :]

            res, info = gmres(
                mat_M, vec_b, x0=guess_spl, tol=tol_gmres, atol=atol_gmres
            )
            if info > 0:
                raise RuntimeError("/!\ GMRES did not converge.")
            elif info < 0:
                raise RuntimeError("Problem with GMRES")

            return t_array, res.reshape((nr_orb, N))

    raise ValueError(f'Method "{method}" not found.')


def solve_quasi_dyson_adapt(
    g_less,
    g_grea,
    t,
    orb,
    orbitals,
    couplings,
    rtol,
    atol,
    start_N=None,
    method="trapz",
    guess=None,
    tol_gmres=1e-10,
    atol_gmres=1e-10,
    max_N=int(1e8),
    verbose=False,
):
    """
    Solve the following equation (for 0 <= u <= t):

    f(u) = g(u - t) - V \int_0^t dv g(u - v) f(v)

    with g(x) = \theta(x) g^>(x) + \theta(-x) g^<(x)

    and returns f.

    TODO: update equation above

    Arguments:
        g_less -- vectorized callable for g^<
        g_grea -- vectorized callable for g^>
        t -- positive float
        orb -- int, orbital index
        orbitals -- 1D array, list of orbital indices
        couplings -- 1D array, list of corresponding couplings
        V -- real or complex parameter
        rtol, atol -- relative and absolute tolerance. Discretization is refined until first one is reached

    Keyword Arguments:
        start_N -- int, starting resolution of discretization.
        method -- one of "cheb", "trapz", "trapz-LU", "trapz-GMRES" (default: {"trapz"})
        guess -- callable t -> 1D array

    Returns:
        f -- callable t -> 1D array
        abs_err -- float, estimated absolute error
        N -- int, number of samples used
    """
    orbitals = np.asarray(orbitals)
    couplings = np.asarray(couplings)

    if start_N is None:
        N = 10
    else:
        N = start_N

    if atol_gmres > atol:
        print("/!\ [Quasi Dyson] atol_gmres is larger than atol!")
    if tol_gmres > rtol:
        print("/!\ [Quasi Dyson] tol_gmres is larger than rtol!")

    times, f_vals = solve_quasi_dyson(
        g_less,
        g_grea,
        t,
        orb,
        orbitals,
        couplings,
        N,
        guess=guess,
        method=method,
        tol_gmres=tol_gmres,
        atol_gmres=atol_gmres,
    )
    f = cpx_interp1d(times, f_vals, axis=1, kind="linear")
    err_times = times

    while True:

        if 2 * N > max_N:
            print(f"/!\ [Quasi Dyson] max discretization reached while err={err}")
            break

        N *= 2

        times, f_vals = solve_quasi_dyson(
            g_less,
            g_grea,
            t,
            orb,
            orbitals,
            couplings,
            N,
            guess=guess,
            method=method,
            tol_gmres=tol_gmres,
            atol_gmres=atol_gmres,
        )
        new_f = cpx_interp1d(times, f_vals, axis=1, kind="linear")

        ff = new_f(err_times)
        err = np.abs(f(err_times) - ff)
        abs_err = np.max(err)
        rel_err = np.max(err / np.abs(ff))

        f = new_f
        err_times = times

        if verbose:
            print(f"{N}: \t abs_err={abs_err}, \t rel err={rel_err}")

        if abs_err < atol or rel_err < rtol:
            break

    return f, abs_err, N


### Cumulated adaptative integral


def cum_int_adapt_simpson(f, xmax, tol=1e-8, maxfeval=100000):
    """
    Adaptative simpson cumulative integral (antiderivative) of `f`, starting from x=0 toward x > 0.

    Arguments:
        f -- real or complex valued callable
        xmax -- antiderivatible is returned on range [0, xmax]

    Keyword Arguments:
        tol -- absolute tolerance in the antiderivative (default: {1e-8})
        maxfeval -- max number of `f` evaluations (default: {100000})

    Returns:
        (coordinates, antiderivative, error) -- three 1D arrays
    """

    a = 0.0
    b = xmax
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
