import numpy as np
from scipy import integrate, linalg, interpolate
from matplotlib import pyplot as plt
import toolbox as tb
from copy import copy
import os

# TODO write test against asymptotic result
# TODO parallelize?
# TODO allow extrapolation
# TODO is real part of C always linear? simplification?
# TODO cleanup notes!

def cheb_points(n, a=-1., b=1.):
    """Chebychev nodes on interval [a, b]"""
    assert(n > 1)
    assert(b > a)
    return (1. - np.cos((np.pi * np.arange(n)) / (n - 1))) * ((b - a) / 2.) + a

def test_cheb_points():
    np.testing.assert_allclose(cheb_points(3), [-1., 0., 1.], atol=1e-15)
    np.testing.assert_allclose(cheb_points(3, -2., 16.), [-2., 7., 16.], atol=1e-15)


### loading Lagrange convolution integrals for the Chebychev method
# lagrange_integrals_less = [None, None]
# lagrange_integrals_grea = [None, None]

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


############## Model ###############

class PhysicsParameters:

    def __init__(self, beta=1., mu_d=0., bias=0., V_cap=1., eps_d=0., eps_c=0., mu_c=0., Gamma=1., U=0.):
        self.beta = beta
        self.mu_d = mu_d # chemical potential on the QD TODO: remove if redondant with eps_d
        self.bias = bias
        self.capac_inv = V_cap # = dV/dQ
        self.eps_d = eps_d # on the QD
        self.eps_c = eps_c # on the QPC
        self.mu_c = mu_c
        self.Gamma = Gamma
        self.U = U
        
        # nr_channels = 4
        # lambda_phi = 1. # = capac ?
        # lambda_chi = 0. # = capac ?
        # ksi_0 = 10.
        
        # def delta_phi(self):
        #     return np.arctan(np.pi * self.lambda_phi * self.Fermi_dos())
        
        # def Fermi_dos(self):
        #     return self.Gamma / (self.eps_d**2 + self.Gamma**2) / np.pi
    
class AccuracyParameters:
    """
    methods available: trapz, cheb
    fft_w_max is in unit of Gamma
    delta_interp_phi: reduce to increase precision. Should be smaller than timescale of variation of bare g.
    """

    def __init__(self, physics_params, time_extrapolate, fft_nr_samples=50000, fft_w_max=50., tol_C=1e-2, slopetol_C=1e-2, delta_interp_phi=0.05, method='trapz'):
        self.PP = copy(physics_params)

        self.time_extrapolate = time_extrapolate
        self.fft_nr_samples = fft_nr_samples
        self.fft_w_max = fft_w_max # in unit of Gamma
        self.tol_C = tol_C
        self.slopetol_C = slopetol_C
        self.delta_interp_phi = delta_interp_phi
        self.method = method


    def omegas_fft(self):
        w_max = self.fft_w_max * self.PP.Gamma
        return np.linspace(-w_max, w_max, tb.misc._next_regular(int(self.fft_nr_samples)))

    def nr_pts_phi(self, t):
        if self.method == 'cheb':
            return int(np.pi * np.sqrt(t / (2 * self.delta_interp_phi)) + 3)
        
        ### default to linear grid
        return int(t / self.delta_interp_phi + 3)

def gen_params(accuracy_params):
    """
    Generator returning (ap, label)
    """

    yield copy(accuracy_params), "original"

    params = copy(accuracy_params)
    params.time_extrapolate *= 2.
    yield params, "time_extrapolate"

    params = copy(accuracy_params)
    params.tol_C *= 4.
    yield params, "tol_C"

    # params = copy(accuracy_params)
    # params.slopetol_C *= 2.
    # yield params, "slopetol_C"

    params = copy(accuracy_params)
    params.delta_interp_phi *= 2.
    yield params, "delta_interp_phi"

    params = copy(accuracy_params)
    params.fft_nr_samples /= 2
    yield params, "fft_nr_samples"

    params = copy(accuracy_params)
    params.fft_w_max /= 2.
    params.fft_nr_samples /= 2
    yield params, "fft_w_max"


class GFModel:

    def __init__(self, physics_params=None, accuracy_params=None):
        self.PP = copy(physics_params) if physics_params is not None else PhysicsParameters()
        self.AP = copy(accuracy_params) if accuracy_params is not None else AccuracyParameters(self.PP, 1.)
    
    def weight(self, Q_up, Q_dn):
        return np.exp(-self.PP.beta * ((self.PP.eps_d - self.PP.mu_d) * (Q_up + Q_dn) + Q_up * Q_dn * self.PP.U))

    def Z_d(self):
        """
        Partition function
        """
        return self.weight(0, 0) + self.weight(1, 0) + self.weight(0, 1) + self.weight(1, 1)
    
    def proba(self, Q_up, Q_dn):
        return self.weight(Q_up, Q_dn) / self.Z_d()
    
    def A_plus(self, Q, t_array):
        raise NotImplementedError

    def A_plus_w(self, Q, nr_freqs):
        raise NotImplementedError
    
    def A_minus(self, Q, t_array):
        raise NotImplementedError
    
    def G_grea(self, t_array):
        prefactor = -1j * np.exp(-1j * t_array * self.PP.eps_d)
        out = self.proba(0, 0) * self.A_plus(0, t_array)
        out += np.exp(-1j * t_array * self.PP.U) * self.proba(0, 1) * self.A_plus(1, t_array) 
        return prefactor * out

    def G_less(self, t_array):
        prefactor = 1j * np.exp(-1j * t_array * self.PP.eps_d)
        out = self.proba(1, 0) * np.conj(self.A_minus(1, t_array))
        out += np.exp(-1j * t_array * self.PP.U) * self.proba(1, 1) * np.conj(self.A_minus(2, t_array))
        return prefactor * out

    def G_grea_NCA_constraint(self, t_array):
        # no U in NCA constraint
        return -1j * np.exp(-1j * t_array * self.PP.eps_d) * self.weight(0, 0) * self.A_plus(0, t_array)

    def G_reta_w_NCA_constraint(self, nr_freqs):
        """
        For NCA in the steady state regime, one only needs the greater quaisparticle GFs in the sector Q=0 (see notes).
        Also, the partition function is reduced to 1.

        Returns: freqs, G_grea, energy shift
        """
        # no U in NCA constraint
        w, A_w, energy_shift = self.A_plus_reta_w(0, nr_freqs)
        return w, -1j * self.weight(0, 0) * A_w, energy_shift - self.PP.eps_d


class NumericModel(GFModel):
    
    def __init__(self, *args, **kwargs):
        super(NumericModel, self).__init__(*args, **kwargs)
        self.N = 3 # nr of different charge states affecting the QPC
        self._cache_g_less_t = [None] * self.N
        self._cache_g_grea_t = [None] * self.N
        self._cache_C_interp = [[None] * self.N, [None] * self.N]
        self._cache_C_tail = [[None] * self.N, [None] * self.N]

    def A_plus(self, Q, times):
        return np.exp(self.C(0, Q, times))
        
    def A_minus(self, Q, times):
        return np.exp(self.C(1, Q, times))
 
    def A_plus_reta_w(self, Q, nr_freqs):
        """
        Returns: freqs, A, energy shift.
        """
        type = 0
        if self._cache_C_tail[type][Q] is None:
            self.compute_C(type, Q)

        slope = self._cache_C_tail[type][Q][1]
        
        times = np.linspace(0, 200. / np.abs(slope.real), nr_freqs)
        C_vals = self.C(type, Q, times)

        # shift energy
        C_vals -= 1j * times * slope.imag

        A = np.exp(C_vals)
        A[0] *= 0.5
        w, A = tb.fourier_transform(times, A)

        return w, A, -slope.imag

    ######## C and phi #######
    
    def C(self, type, Q, times):
        times = np.asarray(times)
        if self._cache_C_interp[type][Q] is None:
            self.compute_C(type, Q)

        C_vals = self._cache_C_interp[type][Q](times)
        mask = np.abs(times) >= self.AP.time_extrapolate

        if mask.any():
            intercept, slope = self._cache_C_tail[type][Q]
            tt = times[mask]
            C_vals[mask] = intercept.real + np.abs(tt) * slope.real + 1j * (np.sign(tt) * intercept.imag + tt * slope.imag)

        return C_vals
    
    def compute_C(self, type, Q):
        """
        Fills cache and returns error estimate.
        """
        if type == 0:
            sign = 1
        elif type == 1:
            sign = -1
        else:
            raise ValueError

        times, C_vals, err = cum_semiinf_adpat_simpson(lambda t: self.phi(sign, Q, t), scale=self.AP.time_extrapolate, tol=self.AP.tol_C, slopetol=self.AP.slopetol_C, extend=False)
        C_vals *= sign * self.PP.capac_inv
        err *= np.abs(self.PP.capac_inv)

        slope = (C_vals[-1] - C_vals[-2]) / (times[-1] - times[-2])
        intercept = C_vals[-1] - slope * times[-1]
        
        C_interp = interpolate.CubicSpline(*tb.symmetrize(times, C_vals, 0., lambda x: np.conj(x)), bc_type='natural', extrapolate=False)
        self._cache_C_interp[type][Q] = C_interp
        self._cache_C_tail[type][Q] = (intercept, slope)

        return err

    def phi(self, sign, Q, t):
        assert(t >= 0.)
        if np.abs(t) < 1e-10:
            return self.g_less_t_fun(Q)(0.)

        times, phi = solve_pseudo_dyson(self.g_less_t_fun(Q), self.g_grea_t_fun(Q), t, sign * self.PP.capac_inv, self.AP.nr_pts_phi(t), method=self.AP.method)
        return phi[-1]
    
    ######## bare GF #########
    
    def delta_leads_R(self, w_array):
        """For both leads"""
        return -1j * self.PP.Gamma * np.ones_like(w_array)
    
    def delta_leads_K(self, w_array):
        """For both leads"""
        return -2j * (tb.fermi(w_array, self.PP.mu_c + 0.5 * self.PP.bias, self.PP.beta) + tb.fermi(w_array, self.PP.mu_c - 0.5 * self.PP.bias, self.PP.beta) - 1.) * np.imag(self.delta_leads_R(w_array))
        
    def g_reta(self, w_array, Q):
        return 1. / (w_array - self.PP.eps_c - Q * self.PP.capac_inv - self.delta_leads_R(w_array))
    
    def g_keld(self, w_array, Q):
        return np.abs(self.g_reta(w_array, Q))**2 * self.delta_leads_K(w_array)
    
    def g_less(self, w_array, Q):
        return np.abs(self.g_reta(w_array, Q))**2 * (0.5 * self.delta_leads_K(w_array) - 1.j * np.imag(self.delta_leads_R(w_array)))
    
    def g_grea(self, w_array, Q):
        return np.abs(self.g_reta(w_array, Q))**2 * (0.5 * self.delta_leads_K(w_array) + 1.j * np.imag(self.delta_leads_R(w_array)))
    
    def g_less_t_fun(self, Q):
        if self._cache_g_less_t[Q] is None:
            g_less = self.g_less(self.AP.omegas_fft(), Q=Q)

            times, g_less_t = tb.inv_fourier_transform(self.AP.omegas_fft(), g_less)
            
            self._cache_g_less_t[Q] = interpolate.CubicSpline(times, g_less_t)
        
        return self._cache_g_less_t[Q]

    def g_grea_t_fun(self, Q):
        if self._cache_g_grea_t[Q] is None:
            g_grea = self.g_grea(self.AP.omegas_fft(), Q=Q)

            times, g_grea_t = tb.inv_fourier_transform(self.AP.omegas_fft(), g_grea)
            
            self._cache_g_grea_t[Q] = interpolate.CubicSpline(times, g_grea_t)

        return self._cache_g_grea_t[Q]


def test_model():
    model = NumericModel()
    model.PP.capac_inv = 0.
    Q = 0

    times = np.linspace(0., 10., 4)
    phi = np.array([model.phi(0, Q, t) for t in times])
    g_less = model.g_less_t_fun(Q)(times)

    idx = np.argmin(np.abs(times))
    assert(times[idx] == 0.)

    np.testing.assert_allclose(phi, g_less[idx])
    
def test_model_methods():
    model = NumericModel()
    model.AP.time_extrapolate = 10.
    times = np.linspace(0., 10., 20)
    Q = 0

    model.AP.method = 'cheb'
    Ap_cheb = model.A_plus(Q, times)

    model.AP.method = 'trapz'
    Ap_trapz = model.A_plus(Q, times)

    np.testing.assert_allclose(Ap_cheb, Ap_trapz, atol=1e-2, rtol=1e-2)

def test_nonreg():
    PP = PhysicsParameters()
    PP.beta = 10.
    PP.bias = 0.
    PP.capac_inv = 1.
    PP.eps_c = 0.0
    PP.eps_d = 0.0
    PP.Gamma = 1.0
    PP.U = 0.0

    AP = AccuracyParameters(PP, time_extrapolate=10.)
    AP.method = 'trapz'
    AP.fft_w_max = 100.
    AP.fft_nr_samples = 100000
    AP.tol_C = 1e-2,
    AP.slopetol_C = 1e-2,
    AP.delta_interp_phi = 0.05

    model = NumericModel(PP, AP)
    times = np.linspace(0., 10., 11)
    C = model.C(0, 0, times)

    C_ref = np.array([ 0.        +0.j        , -0.05929683+0.54617968j,
       -0.1205964 +1.17053087j, -0.15432367+1.81127392j,
       -0.17699975+2.44931755j, -0.19732259+3.08517557j,
       -0.21737896+3.72056224j, -0.23733039+4.35604838j,
       -0.25707505+4.99167134j, -0.27661209+5.62735788j,
       -0.29605259+6.26306652j])

    np.testing.assert_allclose(C, C_ref, rtol=1e-3)


    
if __name__ == "__main__":
    print("Running tests...")
    test_cheb_points()
    test_solve_pseudo_dyson_cheb()
    test_solve_pseudo_dyson_trapz()
    test_cum_semiinf_adpat_simpson()
    test_model()
    test_model_methods()
    test_nonreg()
    print("Success.")