import unittest
import numpy as np
from xrayedge.applications import XRayEdgeSolver, PhysicsParameters, AccuracyParameters


class TestXRayEdgeSolver(unittest.TestCase):
    def test_nonreg_GF(self):
        PP = PhysicsParameters()
        PP.beta = 0.1
        PP.capac_inv = 1.5
        PP.eps_d = 0.0
        PP.bias = 0.0

        AP = AccuracyParameters(PP, time_extrapolate=10.0)
        AP.method = "cheb"
        AP.fft_w_max = 500.0
        AP.fft_nr_samples = 500000
        AP.tol_C = 1e-4
        AP.delta_interp_phi = 0.05

        model = XRayEdgeSolver(PP, AP)
        times = np.linspace(-20.0, 19.0, 10)

        GG = model.G_grea(times)
        GL = model.G_less(times)

        GG_ref = np.array(
            [
                -0.00035666 + 3.87012971e-04j,
                0.00197367 - 1.73820860e-03j,
                -0.01021161 + 7.25880097e-03j,
                0.05000476 - 2.82298611e-02j,
                -0.23194939 + 1.00838350e-01j,
                0.32949644 - 1.15745232e-01j,
                -0.07895729 + 1.89102263e-02j,
                0.01770633 - 2.33368905e-03j,
                -0.00378359 + 1.01992908e-04j,
                0.00076445 + 5.91322966e-05j,
            ]
        )

        GL_ref = np.array(
            [
                -0.00057881 + 0.00058457j,
                0.00050555 - 0.00357701j,
                0.00707893 + 0.01368309j,
                -0.05637395 - 0.03023412j,
                0.25760669 - 0.02325793j,
                -0.30713712 + 0.17273527j,
                0.04167926 - 0.07810862j,
                0.00230112 + 0.02132267j,
                -0.00334141 - 0.00380142j,
                0.00113383 + 0.00024818j,
            ]
        )

        np.testing.assert_allclose(GG, GG_ref, 4)
        np.testing.assert_allclose(GL, GL_ref, 4)


if __name__ == "__main__":
    unittest.main()
