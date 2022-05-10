import unittest
import numpy as np
from numpy import testing
from xrayedge import PhysicsParameters
from xrayedge import reservoir
import toolbox as tb


class TestQPCFrequencyDomain(unittest.TestCase):
    def setUp(self):
        PP = PhysicsParameters()
        PP.beta = 1.0
        PP.D_QPC = 10.0
        PP.eps_QPC = -2.0
        PP.bias_QPC = 0.5
        PP.V_cap = 1.2
        qpc = reservoir.QPC(PP)

        self.PP = PP
        self.QPC = qpc

    def test_gf_reta_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_reta = self.QPC.g_reta(w, Q=0)

        gf_reta_ref = 1.0 / (w + 2.0 + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_reta_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_reta = self.QPC.g_reta(w, Q=Q)

        gf_reta_ref = 1.0 / (w + 2.0 - 1.2 * Q + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_less_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_less = self.QPC.g_less(w, Q=0)

        f_ref = 0.5 * (tb.fermi(w, 0.25, 1.0) + tb.fermi(w, -0.25, 1.0))
        gf_less_ref = -2.0j * f_ref * np.imag(1.0 / (w + 2.0 + 10.0j))

        testing.assert_allclose(gf_less, gf_less_ref, atol=1e-5)

    def test_gf_less_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_less = self.QPC.g_less(w, Q=Q)

        f_ref = 0.5 * (tb.fermi(w, 0.25, 1.0) + tb.fermi(w, -0.25, 1.0))
        gf_less_ref = -2.0j * f_ref * np.imag(1.0 / (w + 2.0 - 1.2 * Q + 10.0j))

        testing.assert_allclose(gf_less, gf_less_ref, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
