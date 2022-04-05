import numpy as np
from scipy import interpolate
import toolbox as tb
from copy import copy
from integral_solvers import solve_pseudo_dyson, cum_semiinf_adpat_simpson

# TODO write test against asymptotic result
# TODO parallelize?
# TODO cleanup notes!


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
    params.time_extrapolate /= 2.
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

def test_nonreg_GF():
    PP = PhysicsParameters()
    PP.beta = 0.1
    PP.capac_inv = 1.5
    PP.eps_d = 0.
    PP.bias = 0.

    AP = AccuracyParameters(PP, time_extrapolate=10.)
    AP.method = 'cheb'
    AP.fft_w_max = 500.
    AP.fft_nr_samples = 500000
    AP.tol_C = 1e-4
    AP.delta_interp_phi = 0.05

    model = NumericModel(PP, AP)
    times = np.linspace(-20., 19., 10)

    GG = model.G_grea(times)
    GL = model.G_less(times)

    GG_ref = np.array([-0.00035666+3.87012971e-04j,  0.00197367-1.73820860e-03j,
       -0.01021161+7.25880097e-03j,  0.05000476-2.82298611e-02j,
       -0.23194939+1.00838350e-01j,  0.32949644-1.15745232e-01j,
       -0.07895729+1.89102263e-02j,  0.01770633-2.33368905e-03j,
       -0.00378359+1.01992908e-04j,  0.00076445+5.91322966e-05j])

    GL_ref = np.array([-0.00057881+0.00058457j,  0.00050555-0.00357701j,
        0.00707893+0.01368309j, -0.05637395-0.03023412j,
        0.25760669-0.02325793j, -0.30713712+0.17273527j,
        0.04167926-0.07810862j,  0.00230112+0.02132267j,
       -0.00334141-0.00380142j,  0.00113383+0.00024818j])

    np.testing.assert_allclose(GG, GG_ref, 4)
    np.testing.assert_allclose(GL, GL_ref, 4)


    
if __name__ == "__main__":
    print("Running tests...")
    test_model()
    test_model_methods()
    test_nonreg()
    test_nonreg_GF()
    print("Success.")