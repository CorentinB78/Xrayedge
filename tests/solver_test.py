import unittest
import numpy as np
import xrayedge as xray


class TestXrayEdgeSolver(unittest.TestCase):

    def test_model(self):
        model = xray.NumericModel()
        model.PP.capac_inv = 0.
        Q = 0

        times = np.linspace(0., 10., 4)
        phi = np.array([model.phi(0, Q, t) for t in times])
        g_less = model.g_less_t_fun(Q)(times)

        idx = np.argmin(np.abs(times))
        assert(times[idx] == 0.)

        np.testing.assert_allclose(phi, g_less[idx])

    def test_model_methods(self):
        model = xray.NumericModel()
        model.AP.time_extrapolate = 10.
        times = np.linspace(0., 10., 20)
        Q = 0

        model.AP.method = 'cheb'
        Ap_cheb = model.A_plus(Q, times)

        model.AP.method = 'trapz'
        Ap_trapz = model.A_plus(Q, times)

        np.testing.assert_allclose(Ap_cheb, Ap_trapz, atol=1e-2, rtol=1e-2)

    def test_nonreg(self):
        PP = xray.PhysicsParameters()
        PP.beta = 10.
        PP.bias = 0.
        PP.capac_inv = 1.
        PP.eps_c = 0.0
        PP.eps_d = 0.0
        PP.Gamma = 1.0
        PP.U = 0.0

        AP = xray.AccuracyParameters(PP, time_extrapolate=10.)
        AP.method = 'trapz'
        AP.fft_w_max = 100.
        AP.fft_nr_samples = 100000
        AP.tol_C = 1e-2,
        AP.delta_interp_phi = 0.05

        model = xray.NumericModel(PP, AP)
        times = np.linspace(0., 10., 11)
        C = model.C(0, 0, times)

        C_ref = np.array([ 0.        +0.j        , -0.05929683+0.54617968j,
        -0.1205964 +1.17053087j, -0.15432367+1.81127392j,
        -0.17699975+2.44931755j, -0.19732259+3.08517557j,
        -0.21737896+3.72056224j, -0.23733039+4.35604838j,
        -0.25707505+4.99167134j, -0.27661209+5.62735788j,
        -0.29605259+6.26306652j])

        np.testing.assert_allclose(C, C_ref, rtol=1e-3)

    def test_nonreg_GF(self):
        PP = xray.PhysicsParameters()
        PP.beta = 0.1
        PP.capac_inv = 1.5
        PP.eps_d = 0.
        PP.bias = 0.

        AP = xray.AccuracyParameters(PP, time_extrapolate=10.)
        AP.method = 'cheb'
        AP.fft_w_max = 500.
        AP.fft_nr_samples = 500000
        AP.tol_C = 1e-4
        AP.delta_interp_phi = 0.05

        model = xray.NumericModel(PP, AP)
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


if __name__ == '__main__':
    unittest.main()