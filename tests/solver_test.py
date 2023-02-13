import unittest
import numpy as np
from numpy import testing
import xrayedge as xray
from copy import copy


class TestCorrelatorSolver(unittest.TestCase):
    def test_no_coupling(self):
        PP = xray.PhysicsParameters()
        AP = xray.AccuracyParameters(PP, 1.0)
        solver = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), [0], [1e-4], AP
        )
        Q = 0

        times = np.linspace(0.0, 10.0, 4)
        phi = np.array([solver.phi(0, Q, t, 0) for t in times])
        g_less = solver.reservoir.g_less_t_fun(Q)(0, 0)(times)

        idx = np.argmin(np.abs(times))
        assert times[idx] == 0.0

        # print(phi - 1e-4 * g_less[idx])

        np.testing.assert_allclose(phi, 1e-4 * g_less[idx], rtol=1e-15, atol=1e-15)

    def test_cheb_vs_trapz(self):
        PP = xray.PhysicsParameters()
        PP.orbitals = [0]
        PP.couplings = [1.5]
        AP = xray.AccuracyParameters(5.0)
        AP.qdyson_rtol = 1e-2

        solver = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), PP.orbitals, PP.couplings, AP
        )
        times = np.linspace(0.0, 10.0, 20)
        Q = 0

        solver.AP.method = "cheb"
        solver.compute_C(0, Q, force_recompute=True)
        Ap_cheb = solver.A_plus(Q, times)

        solver.AP.method = "trapz"
        solver.compute_C(0, Q, force_recompute=True)
        Ap_trapz = solver.A_plus(Q, times)

        np.testing.assert_allclose(Ap_cheb, Ap_trapz, atol=1e-2, rtol=1e-2)

    def test_nonreg(self):
        PP = xray.PhysicsParameters()
        PP.beta = 10.0
        PP.bias_res = 0.0
        PP.couplings = [1.0]
        PP.eps_res = 0.0
        PP.eta_res = 0.0
        PP.eps_sys = 0.0
        PP.Gamma = 1.0
        PP.U = 0.0

        AP = xray.AccuracyParameters(time_extrapolate=10.0)
        AP.method = "trapz"
        AP.tol_C = (1e-2,)

        solver = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), [0], PP.couplings, AP
        )
        times = np.linspace(0.0, 10.0, 11)
        C = solver.C(0, 0, times)

        C_ref = np.array(
            [
                0.0 + 0.0j,
                -0.05930135 - 0.44745357j,
                -0.12060474 - 0.81672744j,
                -0.15431424 - 1.16959136j,
                -0.17696127 - 1.52513533j,
                -0.19738346 - 1.88296818j,
                -0.21747686 - 2.24122385j,
                -0.23735315 - 2.59934597j,
                -0.25702436 - 2.95734156j,
                -0.27655984 - 3.31527777j,
                -0.29602814 - 3.67319077j,
            ]
        )

        np.testing.assert_allclose(C, C_ref, rtol=1e-3)


class TestAPlusReta(unittest.TestCase):
    def test_energy_shift(self):

        PP = xray.PhysicsParameters()
        PP.beta = 1.0
        PP.couplings = [1.0]
        PP.bias_res = 0.0
        PP.eps_res = 0.0
        PP.D_res = 3.0

        AP = xray.AccuracyParameters(time_extrapolate=100.0)
        AP.tol_C = 0.0001
        AP.method = "trapz"
        AP.nr_samples_fft = int(1e6)

        CS = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), [0], PP.couplings, AP
        )

        freqs, A_reta_w, en_shift = CS.A_plus_reta_w(0, 10000)

        times, A_time = xray.fourier.inv_fourier_transform(freqs + en_shift, A_reta_w)
        mask = times > times[-1] / 10.0  # avoid region with Gibbs phenomenon

        testing.assert_allclose(
            A_time[mask], np.exp(CS.C(0, 0, times[mask])), atol=1e-3
        )


class TestCompareWithAnalytic(unittest.TestCase):
    def test(self):
        """
        Compare the slope of Re[C(t)] at long times to exact value.

        An exact analytical result is know at equilibrium, see e.g. Eq.(11) in
        https://doi.org/10.1103/PhysRevLett.79.3740. Here however, the factor 4
        is replaced with 1 (we have only one channel in the QPC) and $\lambda_{\phi}$
        is V_cap / v_F, where v_F is the Fermi velocity. For a 1D chain,
        v_F = 1 / (pi * rho) with rho the density of states at the Fermi level.
        """

        PP = xray.PhysicsParameters()
        PP.beta = 10.0
        PP.couplings = [1.0]
        PP.bias_res = 0.0
        PP.eps_res = 0.0
        PP.D_res = 3.0

        AP = xray.AccuracyParameters(time_extrapolate=100.0)
        AP.tol_C = 0.0001
        AP.method = "trapz"
        AP.nr_samples_fft = int(1e6)

        CS = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), [0], PP.couplings, AP
        )
        tt = np.linspace(0, 100, 1000)
        C_vals = CS.C(0, 0, tt)
        slope_real = (C_vals[-1].real - C_vals[-2].real) / (tt[-1] - tt[-2])

        v_fermi = -1.0 / CS.reservoir.g_reta(np.asarray([0.0]), Q=0)[0].imag
        slope_real_ref = -np.arctan(PP.couplings[0] / v_fermi) ** 2 / (PP.beta * np.pi)

        np.testing.assert_allclose(slope_real, slope_real_ref, 1e-2)


class TestRenormalizedEnergies(unittest.TestCase):
    def test(self):
        """
        When fluctuations can be neglected, the long time slope of Im[C(t)] can be interpreted as a shift in the QD energy levels.

        In the empty QD case, the QD retarded GF is just $G^R(t) = e^{-i E_d t} A^+_0(t)$.
        We also expect to see at long times $G^R(t) = e^{-i (E_d + V_c <n_{QPC}>) t}$, so that
        $A^+_0(t) = e^{-i V_c <n_{QPC}> t}$ at long times.

        In the full (spinless) QD case, we have $G^R(t) = e^{-i E_d t} A^-_1(t)^*$.
        Similarly, we also expect to see at long times $A^-_1(t) = e^{-i V_c <n_{QPC}> t}$.

        We take small V_cap to avoid transmission of QPC fluctuations to the QD.
        """
        PP = xray.PhysicsParameters()
        PP.beta = 10.0
        PP.couplings = [0.1]
        PP.bias_res = 0.0
        PP.eps_res = 0.0
        PP.D_res = 3.0

        AP = xray.AccuracyParameters(time_extrapolate=100.0)
        AP.tol_C = 0.0001
        AP.method = "trapz"
        AP.nr_samples_fft = int(1e6)

        CS = xray.CorrelatorSolver(
            xray.QuantumDot(PP, int(1e4), 100.0), [0], PP.couplings, AP
        )
        res = CS.reservoir
        tt = np.linspace(0, 100, 1000)

        # Q=0, empty QD
        n_QPC = res.occupation(Q=0.0)[0]

        C_vals = CS.C(0, 0, tt)
        slope_imag = (C_vals[-1].imag - C_vals[-2].imag) / (tt[-1] - tt[-2])
        self.assertAlmostEqual(slope_imag, -n_QPC * PP.couplings[0], 2)

        # Q=1, full spinless QD
        n_QPC = res.occupation(Q=1.0)[0]

        C_vals = CS.C(1, 1, tt)
        slope_imag = (C_vals[-1].imag - C_vals[-2].imag) / (tt[-1] - tt[-2])
        self.assertAlmostEqual(slope_imag, n_QPC * PP.couplings[0], 2)


class TestInvarianceTranslation(unittest.TestCase):
    def test(self):
        PP = xray.Parameters()
        PP.beta = 10.0
        PP.bias_res = 0.0
        PP.D_res = 10.0
        PP.eta_res = 0.1
        PP.mu_res = 0.4
        PP.eps_res = [0.0, 0.0, 0.0]
        PP.orbitals = [0]
        PP.couplings = [0.1]

        AP = xray.AccuracyParameters(
            time_extrapolate=20.0, tol_C=1e-4, qdyson_rtol=1e-6
        )

        qpc1 = xray.ExtendedQPC(copy(PP), int(1e4), 100.0)

        PP.orbitals = [1]
        qpc2 = xray.ExtendedQPC(copy(PP), int(1e4), 100.0)

        S1 = xray.CorrelatorSolver(
            qpc1, qpc1.PP.orbitals, qpc1.PP.couplings, AP, verbose=True
        )
        S2 = xray.CorrelatorSolver(
            qpc2, qpc2.PP.orbitals, qpc2.PP.couplings, AP, verbose=True
        )

        times = np.linspace(0, 40, 100)
        cvals1 = S1.C(0, 0, times)
        cvals2 = S2.C(0, 0, times)

        testing.assert_allclose(cvals1, cvals2, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
