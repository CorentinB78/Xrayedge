import toolbox as tb
import numpy as np
from scipy import interpolate, integrate
from copy import copy
from functools import lru_cache


class Reservoir:
    """
    Abstract class for a generic reservoir.
    """

    def __init__(self):
        pass

    def g_less_t(self, Q, orb_a=0, orb_b=0):
        """
        Lesser GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_grea_t(self, Q, orb_a=0, orb_b=0):
        """
        Greater GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    @lru_cache
    def g_less_t_fun(self, Q):
        """
        Lesser GF in times of QPC's central site.

        Returns a (cached) function
        """

        @lru_cache
        def func(orb_a, orb_b):
            times, g_less_t = self.g_less_t(Q, orb_a=orb_a, orb_b=orb_b)
            return interpolate.CubicSpline(times, g_less_t, extrapolate=False)

        return func

    @lru_cache
    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """

        @lru_cache
        def func(orb_a, orb_b):
            times, g_grea_t = self.g_grea_t(Q, orb_a=orb_a, orb_b=orb_b)
            return interpolate.CubicSpline(times, g_grea_t, extrapolate=False)

        return func


class TwoLeadsReservoir(Reservoir):
    """
    Abstract class for a two-lead reservoir. A type of reservoir with a central site coupled to two baths with a chemical potential difference.
    """

    def __init__(
        self, physics_params, nr_samples_fft=None, w_max=None, max_fft_size=int(1e7)
    ):
        super().__init__()
        self.PP = copy(physics_params)

        spreads = np.array([self.PP.D_res, 1.0 / self.PP.beta])
        centers = np.array(
            [
                self.PP.eps_res,
                +0.5 * self.PP.bias_res,
                -0.5 * self.PP.bias_res,
            ]
        )

        if w_max is None:
            w_max = np.max(np.abs(centers[:, None] + spreads[None, :]))
            w_max = 100 * max(
                w_max, np.max(np.abs(centers[:, None] - spreads[None, :]))
            )
        self.w_max = w_max

        dw = np.min(spreads) / 100.0
        ideal_N_fft = int(2 * w_max / dw + 0.5)

        if nr_samples_fft is None:
            self.N_fft = ideal_N_fft
        else:
            self.N_fft = nr_samples_fft

        if self.N_fft >= max_fft_size:
            r = self.N_fft / max_fft_size
            self.w_max = self.w_max / np.sqrt(r)
            dw = dw * np.sqrt(r)
            print(f"/!\ [Reservoir] FFT requires {self.N_fft} grid points.")
            self.N_fft = int(2 * self.w_max / dw + 0.5)
            print(f"Change to {self.N_fft} grid points.")

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together).
        """
        raise NotImplementedError

    def delta_leads_K(self, w_array):
        """
        Keldysh hybridization function in frequencies for left and right leads (together).
        """
        return (
            -2j
            * (
                tb.fermi(w_array, 0.5 * self.PP.bias_res, self.PP.beta)
                + tb.fermi(w_array, -0.5 * self.PP.bias_res, self.PP.beta)
                - 1.0
            )
            * np.imag(self.delta_leads_R(w_array))
        )

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of central site.
        """
        raise NotImplementedError

    def g_keld(self, w_array, Q):
        """
        Keldysh GF in frequencies of central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * self.delta_leads_K(w_array)

    def g_less(self, w_array, Q):
        """
        Lesser GF in frequencies of central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            - 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_grea(self, w_array, Q):
        """
        Greater GF in frequencies of central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            + 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_less_t(self, Q, orb_a=0, orb_b=0):
        """
        Lesser GF in times of contact site.
        """
        if orb_a != 0 or orb_b != 0:
            raise ValueError(
                "A TwoLeadsReservoir can be evaluated only at the central site (orbital=0)"
            )
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_less = self.g_less(w, Q=Q)

        times, g_less_t = tb.inv_fourier_transform(w, g_less)
        return times, g_less_t

    def g_grea_t(self, Q, orb_a=0, orb_b=0):
        """
        Greater GF in times of contact site.
        """
        if orb_a != 0 or orb_b != 0:
            raise ValueError(
                "A TwoLeadsReservoir can be evaluated only at the central site (orbital=0)"
            )
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_grea = self.g_grea(w, Q=Q)

        times, g_grea_t = tb.inv_fourier_transform(w, g_grea)
        return times, g_grea_t

    def occupation(self, Q):
        w, dw = np.linspace(-self.w_max, self.w_max, self.N_fft, retstep=True)
        g_less = self.g_less(w, Q=Q)

        return integrate.simpson(y=g_less.imag, dx=dw) / (2 * np.pi)

    def transmission(self, w_array, Q):
        return self.delta_leads_R(w_array).imag * self.g_reta(w_array, Q).imag


class QuantumDot(TwoLeadsReservoir):
    """
    Quantum Dot. A type of reservoir with a central site coupled to two baths of infintie bandwidth and with a chemical potential difference.

    Provides real time Green functions at the central site for different charge offsets Q.
    """

    def __init__(self, *args, **kwargs):
        """
        Arguments:
            physics_params -- a class with parameters `D_res`, `eps_res`, `bias_res`, `beta` and `V_cap`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT (default: {None} which auto determine an optimal value)
            w_max -- max frequency for FFT (default: {None} which auto determine an optimal value)
        """
        super().__init__(*args, **kwargs)

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together).
        """
        return -1j * self.PP.D_res * np.ones_like(w_array)

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of central site.
        """
        w_array = np.asarray(w_array)
        return 1.0 / (
            w_array - self.PP.eps_res - Q * self.PP.V_cap - self.delta_leads_R(w_array)
        )


class QPC(TwoLeadsReservoir):
    """
    Quantum Point Contact. A type of reservoir with a central site coupled to two semicircular baths and with a chemical potential difference.

    Provides real time Green functions at the central site for different charge offsets Q.
    """

    def __init__(self, *args, **kwargs):
        """
        Arguments:
            physics_params -- a class with parameters `D_res`, `eps_res`, `mu_res`, `eta_res`, `bias_res`, `beta` and `V_cap`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT (default: {None} which auto determine an optimal value)
            w_max -- max frequency for FFT (default: {None} which auto determine an optimal value)
        """
        super().__init__(*args, **kwargs)

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together).
        """
        w_array = np.asarray(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = tb.semicirc_retarded_gf(t)
        return (
            2.0
            * t**2
            * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)
        )

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of central site.
        """
        w_array = np.asarray(w_array)
        return 1.0 / (
            w_array
            + self.PP.mu_res
            + self.PP.eta_res * 1j
            - self.PP.eps_res
            - self.PP.D_res
            - Q * self.PP.V_cap
            - self.delta_leads_R(w_array)
        )


class OneDChain(Reservoir):
    def __init__(self, physics_params, tmax, nr_times):
        """
        _summary_

        Arguments:
            physics_params -- a class with parameters `D_res`, `eps_res`, `mu_res`, `bias_res`, `beta`
            tmax -- max of the time grid on which to return Green functions
            nr_times -- nr of points in half the time grid
        """
        super().__init__()
        self.PP = copy(physics_params)
        self.tmax = tmax
        self.nr_times = nr_times

    def stat(self, e):
        fL = tb.fermi(e, self.PP.mu_res - self.PP.bias_res / 2.0, self.PP.beta)
        fR = tb.fermi(e, self.PP.mu_res + self.PP.bias_res / 2.0, self.PP.beta)
        return (fL + fR) / 2.0

    def one_minus_stat(self, e):
        fL = tb.fermi(e, self.PP.mu_res - self.PP.bias_res / 2.0, -self.PP.beta)
        fR = tb.fermi(e, self.PP.mu_res + self.PP.bias_res / 2.0, -self.PP.beta)
        return (fL + fR) / 2.0

    def energy(self, k):
        return self.PP.eps_res - self.PP.D_res * np.cos(k)

    def g_less_t(self, Q, orb_a=0, orb_b=0):
        """
        Lesser GF in times.
        """
        if Q != 0.0:
            raise ValueError("Q != 0 has not been implemented yet.")

        times = np.linspace(-self.tmax, self.tmax, 2 * self.nr_times + 1)
        out = np.empty_like(times, dtype=complex)

        for i, t in enumerate(times):

            def f_re(k):
                e = self.energy(k)
                dn = orb_a - orb_b
                return self.stat(e) * np.cos(k * dn - e * t)

            def f_im(k):
                e = self.energy(k)
                dn = orb_a - orb_b
                return self.stat(e) * np.sin(k * dn - e * t)

            out[i] = integrate.quad(f_re, -np.pi, np.pi)[0]
            out[i] += 1j * integrate.quad(f_im, -np.pi, np.pi)[0]

        return times, out * 1j / (2 * np.pi)

    def g_grea_t(self, Q, orb_a=0, orb_b=0):
        """
        Greater GF in times.
        """
        if Q != 0.0:
            raise ValueError("Q != 0 has not been implemented yet.")

        times = np.linspace(-self.tmax, self.tmax, 2 * self.nr_times + 1)
        out = np.empty_like(times, dtype=complex)

        for i, t in enumerate(times):

            def f_re(k):
                e = self.energy(k)
                dn = orb_a - orb_b
                return self.one_minus_stat(e) * np.cos(k * dn - e * t)

            def f_im(k):
                e = self.energy(k)
                dn = orb_a - orb_b
                return self.one_minus_stat(e) * np.sin(k * dn - e * t)

            out[i] = integrate.quad(f_re, -np.pi, np.pi)[0]
            out[i] += 1j * integrate.quad(f_im, -np.pi, np.pi)[0]

        return times, -out * 1j / (2 * np.pi)
