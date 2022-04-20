import toolbox as tb
import numpy as np
from scipy import interpolate
from copy import copy


class Reservoir:
    def __init__(self):
        """
        Abstract class for a generic reservoir.
        """
        self.N = 3  # nr of different charge states affecting the QPC

        self._cache_g_less_t = [None] * self.N
        self._cache_g_grea_t = [None] * self.N

    def g_less_t(self, Q):
        """
        Lesser GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_grea_t(self, Q):
        """
        Greater GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_less_t_fun(self, Q):
        """
        Lesser GF in times of QPC's central site.

        Returns a (cached) function
        """
        if self._cache_g_less_t[Q] is None:
            times, g_less_t = self.g_less_t(Q)
            self._cache_g_less_t[Q] = interpolate.CubicSpline(times, g_less_t)

        return self._cache_g_less_t[Q]

    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """
        if self._cache_g_grea_t[Q] is None:
            times, g_grea_t = self.g_grea_t(Q)
            self._cache_g_grea_t[Q] = interpolate.CubicSpline(times, g_grea_t)

        return self._cache_g_grea_t[Q]


class QPC(Reservoir):
    def __init__(self, physics_params, accuracy_params):
        """
        Quantum Point Contact. A type of reservoir with a central site coupled to two baths with a chemical potential difference.

        Provides real time Green functions at the central site for different charge offsets Q.

        Arguments:
            physics_params -- a `PhysicsParameters` instance
            accuracy_params -- a `AccuracyParameters` instance
        """
        super().__init__()
        self.PP = copy(physics_params)
        self.AP = copy(accuracy_params)

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together) of QPC.
        """
        return -1j * self.PP.Gamma * np.ones_like(w_array)

    def delta_leads_K(self, w_array):
        """
        Keldysh hybridization function in frequencies for left and right leads (together) of QPC.
        """
        return (
            -2j
            * (
                tb.fermi(w_array, self.PP.mu_c + 0.5 * self.PP.bias, self.PP.beta)
                + tb.fermi(w_array, self.PP.mu_c - 0.5 * self.PP.bias, self.PP.beta)
                - 1.0
            )
            * np.imag(self.delta_leads_R(w_array))
        )

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of QPC's central site.
        """
        return 1.0 / (
            w_array
            - self.PP.eps_c
            - Q * self.PP.capac_inv
            - self.delta_leads_R(w_array)
        )

    def g_keld(self, w_array, Q):
        """
        Keldysh GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * self.delta_leads_K(w_array)

    def g_less(self, w_array, Q):
        """
        Lesser GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            - 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_grea(self, w_array, Q):
        """
        Greater GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            + 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_less_t(self, Q):
        """
        Lesser GF in times of contact site.
        """
        g_less = self.g_less(self.AP.omegas_fft(), Q=Q)

        times, g_less_t = tb.inv_fourier_transform(self.AP.omegas_fft(), g_less)
        return times, g_less_t

    def g_grea_t(self, Q):
        """
        Greater GF in times of contact site.
        """
        g_grea = self.g_grea(self.AP.omegas_fft(), Q=Q)

        times, g_grea_t = tb.inv_fourier_transform(self.AP.omegas_fft(), g_grea)
        return times, g_grea_t
