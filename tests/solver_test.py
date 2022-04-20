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
        AP = xray.AccuracyParameters(PP, 10.0)
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
        PP.bias = 0.0
        PP.capac_inv = 1.0
        PP.eps_c = 0.0
        PP.eps_d = 0.0
        PP.Gamma = 1.0
        PP.U = 0.0

        AP = xray.AccuracyParameters(PP, time_extrapolate=10.0)
        AP.method = "trapz"
        AP.tol_C = (1e-2,)
        AP.delta_interp_phi = 0.05

        solver = xray.CorrelatorSolver(xray.QPC(PP), PP.capac_inv, AP)
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


if __name__ == "__main__":
    unittest.main()
