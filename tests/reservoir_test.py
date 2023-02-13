import unittest
import numpy as np
from numpy import testing
from xrayedge import PhysicsParameters
from xrayedge import reservoir
import matplotlib.pyplot as plt
from scipy import integrate


class TestQuantumDotFrequencyDomain(unittest.TestCase):
    def setUp(self):
        PP = PhysicsParameters()
        PP.beta = 1.0
        PP.D_res = 10.0
        PP.eps_res = -2.0
        PP.bias_res = 0.5
        PP.orbitals = [0]
        PP.couplings = [1.2]
        PP.eta_res = 0.0
        res = reservoir.QuantumDot(PP, int(1e4), 100.0)

        self.PP = PP
        self.res = res

    def test_gf_reta_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_reta = self.res.g_reta(w, Q=0)[:, 0, 0]

        gf_reta_ref = 1.0 / (w + 2.0 + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_reta_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_reta = self.res.g_reta(w, Q=Q)[:, 0, 0]

        gf_reta_ref = 1.0 / (w + 2.0 - 1.2 * Q + 10.0j)

        testing.assert_allclose(gf_reta, gf_reta_ref, atol=1e-5)

    def test_gf_less_0(self):
        w = np.linspace(-50, 50, 1000)
        gf_less = self.res.g_less(w, Q=0)[:, 0, 0]

        f_ref = 0.5 * (reservoir.fermi(w, 0.25, 1.0) + reservoir.fermi(w, -0.25, 1.0))
        gf_less_ref = -2.0j * f_ref * np.imag(1.0 / (w + 2.0 + 10.0j))

        testing.assert_allclose(gf_less, gf_less_ref, atol=1e-5)

    def test_gf_less_1(self):
        w = np.linspace(-50, 50, 1000)
        Q = 0.5
        gf_less = self.res.g_less(w, Q=Q)[:, 0, 0]

        f_ref = 0.5 * (reservoir.fermi(w, 0.25, 1.0) + reservoir.fermi(w, -0.25, 1.0))
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
        PP.orbitals = [0]
        PP.couplings = [1.2]
        res = reservoir.QPC(PP, int(1e4), 100.0)

        self.PP = PP
        self.res = res

    def test_gf_reta(self):
        w = np.linspace(-30, 30, 1000)
        gf_reta = self.res.g_reta(w, Q=0)[:, 0, 0]

        mask = w < 0.0
        testing.assert_allclose(gf_reta[mask].imag, 0.0)

        mask = np.logical_and(w >= 0, w <= 20.0)
        gf_reta_ref = 1.0 / (-5.0 + 1j * np.sqrt(10.0**2 - (w - 10.0) ** 2))
        testing.assert_allclose(gf_reta[mask], gf_reta_ref[mask], atol=1e-5)

        mask = w > 20.0
        testing.assert_allclose(gf_reta[mask].imag, 0.0)

    def test_transmission(self):
        w = np.linspace(-10, 10, 1000)
        transm = self.res.transmission(w, Q=0)

        mask = w < 0.0
        testing.assert_allclose(transm[mask], 0.0)

        mask = np.logical_and(w >= 0, w <= 20.0)
        transm_ref = (20.0 - w) * w / (5.0**2 + (20.0 - w) * w)
        testing.assert_allclose(transm[mask], transm_ref[mask], atol=1e-5)


class TestQPCFrequencyDomainEta(unittest.TestCase):
    def setUp(self):
        PP = PhysicsParameters()
        PP.beta = 10.0
        PP.D_res = 20.0
        PP.eps_res = 0.2
        PP.eta_res = 1.0
        PP.bias_res = 0.0
        PP.orbitals = [0]
        PP.couplings = [0.1]
        res = reservoir.QPC(PP, int(1e4), 100.0)

        self.PP = PP
        self.res = res

    def test_gf_reta_fermi_level(self):
        gf_0 = self.res.g_reta(0.0, Q=0)
        gf_1 = self.res.g_reta(1e-5, Q=0)
        gf_2 = self.res.g_reta(-1e-5, Q=0)

        testing.assert_allclose(gf_0, gf_1, atol=1e-4)
        testing.assert_allclose(gf_0, gf_2, atol=1e-4)

        gf_0 = self.res.delta_leads_R_left(0.0)
        gf_1 = self.res.delta_leads_R_left(1e-5)
        gf_2 = self.res.delta_leads_R_left(-1e-5)

        testing.assert_allclose(gf_0, gf_1, atol=1e-4)
        testing.assert_allclose(gf_0, gf_2, atol=1e-4)


class TestExtendedQPC(unittest.TestCase):
    def test(self):
        PP = PhysicsParameters()
        PP.D_res = 5.0
        PP.mu_res = 5.0
        PP.eta_res = 0.0
        PP.beta = 1.0

        N = 11
        PP.eps_res = np.zeros(N)
        PP.orbitals = [3, 4, 5, 6]
        PP.couplings = [0.5, 1.0, 1.0, 0.5]

        qpc = reservoir.ExtendedQPC(PP, int(1e4), 100.0)

        w = np.linspace(-10, 10, int(1e5))
        gr = qpc.g_reta(w, Q=0)
        ga = np.conj(gr).swapaxes(1, 2)
        norm = integrate.simpson(x=w, y=gr - ga, axis=0)
        testing.assert_allclose(norm, -2j * np.pi * np.eye(N), atol=1e-2)

        print(qpc.occupation(Q=0))

        # plt.plot(qpc.occupation(Q=0))
        # plt.plot(qpc.occupation(Q=1), "--")
        # plt.show()


if __name__ == "__main__":
    unittest.main()
