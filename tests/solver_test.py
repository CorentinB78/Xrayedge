import unittest
import numpy as np
import xrayedge as xray


class TestCorrelatorSolver(unittest.TestCase):
    def test_no_coupling(self):
        PP = xray.PhysicsParameters()
        AP = xray.AccuracyParameters(PP, 1.0)
        solver = xray.CorrelatorSolver(xray.QPC(PP), 0.0, AP)
        Q = 0

        times = np.linspace(0.0, 10.0, 4)
        phi = np.array([solver.phi(0, Q, t) for t in times])
        g_less = solver.reservoir.g_less_t_fun(Q)(times)

        idx = np.argmin(np.abs(times))
        assert times[idx] == 0.0

        np.testing.assert_allclose(phi, g_less[idx])

    def test_cheb_vs_trapz(self):
        PP = xray.PhysicsParameters()
        AP = xray.AccuracyParameters(10.0)
        solver = xray.CorrelatorSolver(xray.QPC(PP), 1.5, AP)
        times = np.linspace(0.0, 10.0, 20)
        Q = 0

        solver.AP.method = "cheb"
        Ap_cheb = solver.A_plus(Q, times)

        solver.AP.method = "trapz"
        Ap_trapz = solver.A_plus(Q, times)

        np.testing.assert_allclose(Ap_cheb, Ap_trapz, atol=1e-2, rtol=1e-2)

    def test_nonreg(self):
        PP = xray.PhysicsParameters()
        PP.beta = 10.0
        PP.bias_QPC = 0.0
        PP.V_cap = 1.0
        PP.eps_QPC = 0.0
        PP.eps_QD = 0.0
        PP.Gamma = 1.0
        PP.U = 0.0

        AP = xray.AccuracyParameters(time_extrapolate=10.0)
        AP.method = "trapz"
        AP.tol_C = (1e-2,)
        AP.delta_interp_phi = 0.05

        solver = xray.CorrelatorSolver(xray.QPC(PP), PP.V_cap, AP)
        times = np.linspace(0.0, 10.0, 11)
        C = solver.C(0, 0, times)

        C_ref = np.array(
            [
                0.0 + 0.0j,
                -0.05929683 + 0.54617968j,
                -0.1205964 + 1.17053087j,
                -0.15432367 + 1.81127392j,
                -0.17699975 + 2.44931755j,
                -0.19732259 + 3.08517557j,
                -0.21737896 + 3.72056224j,
                -0.23733039 + 4.35604838j,
                -0.25707505 + 4.99167134j,
                -0.27661209 + 5.62735788j,
                -0.29605259 + 6.26306652j,
            ]
        )

        np.testing.assert_allclose(C, C_ref, rtol=1e-3)


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
        PP.V_cap = 1.0
        PP.bias_QPC = 0.0
        PP.eps_QPC = 0.0
        PP.mu_QPC = 0.0
        PP.D_QPC = 3.0

        AP = xray.AccuracyParameters(time_extrapolate=100.0)
        AP.tol_C = 0.0001
        AP.delta_interp_phi = 0.01
        AP.method = "trapz"
        AP.nr_samples_fft = int(1e6)

        CS = xray.CorrelatorSolver(xray.QPC(PP), PP.V_cap, AP)
        tt = np.linspace(0, 100, 1000)
        C_vals = CS.C(0, 0, tt)
        slope_real = (C_vals[-1].real - C_vals[-2].real) / (tt[-1] - tt[-2])

        v_fermi = -1.0 / CS.reservoir.g_reta(np.asarray([PP.mu_QPC]), Q=0)[0].imag
        slope_real_ref = -np.arctan(PP.V_cap / v_fermi) ** 2 / (PP.beta * np.pi)

        self.assertAlmostEqual(slope_real, slope_real_ref, 2)


if __name__ == "__main__":
    unittest.main()
