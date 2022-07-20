import toolbox as tb
import numpy as np
from scipy import interpolate, integrate
from copy import copy


class Reservoir:
    """
    Abstract class for a generic reservoir.
    """

    def __init__(self):
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
            self._cache_g_less_t[Q] = interpolate.CubicSpline(
                times, g_less_t, extrapolate=False
            )

        return self._cache_g_less_t[Q]

    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """
        if self._cache_g_grea_t[Q] is None:
            times, g_grea_t = self.g_grea_t(Q)
            self._cache_g_grea_t[Q] = interpolate.CubicSpline(
                times, g_grea_t, extrapolate=False
            )

        return self._cache_g_grea_t[Q]


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

    def g_less_t(self, Q):
        """
        Lesser GF in times of contact site.
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_less = self.g_less(w, Q=Q)

        times, g_less_t = tb.inv_fourier_transform(w, g_less)
        return times, g_less_t

    def g_grea_t(self, Q):
        """
        Greater GF in times of contact site.
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_grea = self.g_grea(w, Q=Q)

        times, g_grea_t = tb.inv_fourier_transform(w, g_grea)
        return times, g_grea_t

    def occupation(self, Q):
        w, dw = np.linspace(-self.w_max, self.w_max, self.N_fft, retstep=True)
        g_less = self.g_less(w, Q=Q)

        return integrate.simpson(y=g_less.imag, dx=dw) / (2 * np.pi)


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
            physics_params -- a class with parameters `D_res`, `eps_res`, `eta_res`, `bias_res`, `beta` and `V_cap`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT (default: {None} which auto determine an optimal value)
            w_max -- max frequency for FFT (default: {None} which auto determine an optimal value)
        """
        super().__init__(*args, **kwargs)

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together).
        """
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = tb.semicirc_retarded_gf(t)
        return 2.0 * t**2 * sc_gf(w_array - 2.0 * t + self.PP.eta_res * 1j)

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of central site.
        """
        w_array = np.asarray(w_array)
        return 1.0 / (
            w_array
            + self.PP.eta_res * 1j
            - self.PP.eps_res
            - self.PP.D_res
            - Q * self.PP.V_cap
            - self.delta_leads_R(w_array)
        )
