import unittest
import numpy as np
from numpy import testing
from xrayedge import PhysicsParameters
from xrayedge import reservoir
import toolbox as tb
import matplotlib.pyplot as plt


class TestQuantumDotFrequencyDomain(unittest.TestCase):
    def setUp(self):
        PP = PhysicsParameters()
        PP.beta = 1.0
        PP.D_res = 10.0
        PP.eps_res = -2.0
        PP.bias_res = 0.5
        PP.V_cap = 1.2
        res = reservoir.QuantumDot(PP)

        self.PP = PP
        self.res = res

    def test_gf_reta_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_reta = self.res.g_reta(w, Q=0)

        gf_reta_ref = 1.0 / (w + 2.0 + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_reta_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_reta = self.res.g_reta(w, Q=Q)

        gf_reta_ref = 1.0 / (w + 2.0 - 1.2 * Q + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_less_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_less = self.res.g_less(w, Q=0)

        f_ref = 0.5 * (tb.fermi(w, 0.25, 1.0) + tb.fermi(w, -0.25, 1.0))
        gf_less_ref = -2.0j * f_ref * np.imag(1.0 / (w + 2.0 + 10.0j))

        testing.assert_allclose(gf_less, gf_less_ref, atol=1e-5)

    def test_gf_less_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_less = self.res.g_less(w, Q=Q)

        f_ref = 0.5 * (tb.fermi(w, 0.25, 1.0) + tb.fermi(w, -0.25, 1.0))
        gf_less_ref = -2.0j * f_ref * np.imag(1.0 / (w + 2.0 - 1.2 * Q + 10.0j))

        testing.assert_allclose(gf_less, gf_less_ref, atol=1e-5)


class TestQPCFrequencyDomain(unittest.TestCase):
    def setUp(self):
        PP = PhysicsParameters()
        PP.beta = 1.0
        PP.D_res = 10.0
        PP.eps_res = 5.0
        PP.eta_res = 0.0
        PP.bias_res = 0.5
        PP.V_cap = 1.2
        res = reservoir.QPC(PP)

        self.PP = PP
        self.res = res

    def test_gf_reta(self):
        w = np.linspace(-30, 30, 1000)
        gf_reta = self.res.g_reta(w, Q=0)

        mask = w < 0.0
        testing.assert_allclose(gf_reta[mask].imag, 0.0)

        mask = np.logical_and(w >= 0, w <= 20.0)
        gf_reta_ref = 1.0 / (-5.0 + 1j * np.sqrt(10.0**2 - (w - 10.0) ** 2))
        testing.assert_allclose(gf_reta[mask], gf_reta_ref[mask], atol=1e-5)

        mask = w > 20.0
        testing.assert_allclose(gf_reta[mask].imag, 0.0)


if __name__ == "__main__":
    unittest.main()
