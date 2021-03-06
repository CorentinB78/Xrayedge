import unittest
import numpy as np
from scipy import interpolate, integrate
import xrayedge as xray
from xrayedge.integral_solvers import (
    cheb_points,
    QuasiToeplitzMatrix,
    solve_quasi_dyson,
    solve_quasi_dyson_last_time,
)


class TestQuasiToeplitzMatrix(unittest.TestCase):
    def test_matvec_product(self):
        c = np.array([1.0j, 2.0, -3.0])
        r = np.array([-2.0, 3.0j, 1.0])
        corr_0 = np.array([0.5, -0.5j, 0.0])
        corr_1 = np.array([1.5, 0.5j, 1.0])
        b = np.array([2.0, 5.0j, -3.0])

        M = QuasiToeplitzMatrix(c, r, (corr_0, corr_1))

        M_ref = np.array(
            [
                [1.0j + 0.5, 3.0j, 2.5],
                [2.0 - 0.5j, 1.0j, 3.5j],
                [-3.0, 2.0, 1.0 + 1.0j],
            ]
        )

        np.testing.assert_array_almost_equal(M @ b, np.dot(M_ref, b))


class TestChebPoints(unittest.TestCase):
    def test_cheb_points(self):
        np.testing.assert_allclose(cheb_points(3), [-1.0, 0.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(
            cheb_points(3, -2.0, 16.0), [-2.0, 7.0, 16.0], atol=1e-15
        )


class TestSolvePseudoDyson(unittest.TestCase):
    @property
    def solution_ref(self):
        x = np.array(
            [
                0.0,
                0.00308191,
                0.01231498,
                0.02766126,
                0.04905771,
                0.07641638,
                0.10962486,
                0.1485467,
                0.19302194,
                0.24286784,
                0.29787957,
                0.35783106,
                0.42247597,
                0.49154866,
                0.5647653,
                0.64182501,
                0.72241115,
                0.80619256,
                0.89282499,
                0.98195242,
                1.07320862,
                1.1662186,
                1.26060016,
                1.35596546,
                1.45192263,
                1.54807737,
                1.64403454,
                1.73939984,
                1.8337814,
                1.92679138,
                2.01804758,
                2.10717501,
                2.19380744,
                2.27758885,
                2.35817499,
                2.4352347,
                2.50845134,
                2.57752403,
                2.64216894,
                2.70212043,
                2.75713216,
                2.80697806,
                2.8514533,
                2.89037514,
                2.92358362,
                2.95094229,
                2.97233874,
                2.98768502,
                2.99691809,
                3.0,
            ]
        )

        # computed with Cheb 50 points, accuracy estimated < 1e-4
        y = np.array(
            [
                -3.56227993 + 0.0j,
                -3.53858641 + 0.0j,
                -3.46866995 + 0.0j,
                -3.35592324 + 0.0j,
                -3.20568068 + 0.0j,
                -3.02478108 + 0.0j,
                -2.82103077 + 0.0j,
                -2.6026321 + 0.0j,
                -2.37763213 + 0.0j,
                -2.15345221 + 0.0j,
                -1.93652033 + 0.0j,
                -1.73204687 + 0.0j,
                -1.54390907 + 0.0j,
                -1.37454711 + 0.0j,
                -1.22576334 + 0.0j,
                -1.09743128 + 0.0j,
                -0.98923376 + 0.0j,
                -0.90005678 + 0.0j,
                -0.82839417 + 0.0j,
                -0.77249059 + 0.0j,
                -0.73051624 + 0.0j,
                -0.70067775 + 0.0j,
                -0.68116526 + 0.0j,
                -0.67050317 + 0.0j,
                -0.6672488 + 0.0j,
                -0.67016811 + 0.0j,
                -0.67819753 + 0.0j,
                -0.69042278 + 0.0j,
                -0.70608382 + 0.0j,
                -0.72446046 + 0.0j,
                -0.74504178 + 0.0j,
                -0.76730576 + 0.0j,
                -0.79082084 + 0.0j,
                -0.81518871 + 0.0j,
                -0.8400582 + 0.0j,
                -0.86508789 + 0.0j,
                -0.88984925 + 0.0j,
                -0.91443172 + 0.0j,
                -0.93813583 + 0.0j,
                -0.96084053 + 0.0j,
                -0.98227983 + 0.0j,
                -1.0022094 + 0.0j,
                -1.02039572 + 0.0j,
                -1.0366264 + 0.0j,
                -1.05070782 + 0.0j,
                -1.06247138 + 0.0j,
                -1.07177402 + 0.0j,
                -1.07850205 + 0.0j,
                -1.08257246 + 0.0j,
                -1.08393491 + 0.0j,
            ]
        )

        return interpolate.BarycentricInterpolator(x, y)

    def test_cheb(self):
        V = 2.0
        t = 3.0

        def cst_func(c):
            return np.vectorize(lambda x: c)

        time, phi = solve_quasi_dyson(
            cst_func(1.0), cst_func(1.0), t, V, 10, method="cheb"
        )
        np.testing.assert_allclose(phi, 1.0 / (1.0 + V * t))

        time, phi = solve_quasi_dyson(
            cst_func(0.0), cst_func(1.0), t, V, 10, method="cheb"
        )
        np.testing.assert_allclose(phi, 0.0)

        time, phi = solve_quasi_dyson(
            cst_func(1.0), cst_func(0.0), t, V, 20, method="cheb"
        )
        np.testing.assert_allclose(phi, np.exp(V * (time - t)))

        time, phi = solve_quasi_dyson(np.sin, np.cos, t, V, 50, method="cheb")
        np.testing.assert_allclose(phi, self.solution_ref(time), atol=1e-4)

    def test_trapz_auto_refine(self):
        V = 2.0
        t = 3.0

        phi_t, err, N = solve_quasi_dyson_last_time(
            np.sin,
            np.cos,
            t,
            V,
            rtol=1e-3,
            atol=1e-3,
            method="trapz",
            verbose=False,
        )
        np.testing.assert_allclose(phi_t, self.solution_ref(t), atol=1e-3)

    def test_trapz(self):
        V = 2.0
        t = 3.0

        def cst_func(c):
            return np.vectorize(lambda x: c)

        time, phi = solve_quasi_dyson(
            cst_func(1.0), cst_func(1.0), t, V, 10, method="trapz"
        )
        np.testing.assert_allclose(phi, 1.0 / (1.0 + V * t))

        time, phi = solve_quasi_dyson(
            cst_func(0.0), cst_func(1.0), t, V, 10, method="trapz"
        )
        np.testing.assert_allclose(phi, 0.0)

        time, phi = solve_quasi_dyson(
            cst_func(1.0), cst_func(0.0), t, V, 1000, method="trapz"
        )
        np.testing.assert_allclose(phi, np.exp(V * (time - t)), rtol=1e-4, atol=1e-4)

        time, phi = solve_quasi_dyson(np.sin, np.cos, t, V, 100, method="trapz")
        np.testing.assert_allclose(phi, self.solution_ref(time), atol=1e-3)

    def test_second_order_cheb(self):
        gl = lambda x: np.sin(1.5 * x) * np.exp(-((x - 1.0) ** 2) / 3.0)
        gg = lambda x: np.cos(x + 2.0) * np.exp(-((x + 0.5) ** 2) / 2.0)
        t = 3.0
        V = 0.0001

        times, f_vals = solve_quasi_dyson(gl, gg, t, V, 50, method="cheb")

        ### perturbation orders in V
        f0 = gl(times - t)
        f1 = np.array(
            [
                np.array(integrate.quad(lambda x: gg(u - x) * gl(x - t), 0, u)[:2])
                + np.array(integrate.quad(lambda x: gl(u - x) * gl(x - t), u, t)[:2])
                for u in times
            ]
        )
        f1_err = f1[:, 1]
        f1 = -f1[:, 0]

        np.testing.assert_array_less(f1_err, 1e-8)
        np.testing.assert_allclose(f_vals, f0 + V * f1, atol=1e-8)
        np.testing.assert_allclose((f_vals - f0) / V, f1, atol=1e-5)

    def test_second_order_trapz(self):
        gl = lambda x: np.sin(1.5 * x) * np.exp(-((x - 1.0) ** 2) / 3.0)
        gg = lambda x: np.cos(x + 2.0) * np.exp(-((x + 0.5) ** 2) / 2.0)
        t = 3.0
        V = 0.0001

        times, f_vals = solve_quasi_dyson(gl, gg, t, V, 200, method="trapz")

        ### perturbation orders in V
        f0 = gl(times - t)
        f1 = np.array(
            [
                np.array(integrate.quad(lambda x: gg(u - x) * gl(x - t), 0, u)[:2])
                + np.array(integrate.quad(lambda x: gl(u - x) * gl(x - t), u, t)[:2])
                for u in times
            ]
        )
        f1_err = f1[:, 1]
        f1 = -f1[:, 0]

        np.testing.assert_array_less(f1_err, 1e-8)
        np.testing.assert_allclose(f_vals, f0 + V * f1, atol=1e-8)
        np.testing.assert_allclose((f_vals - f0) / V, f1, atol=1e-5)


class TestCumAdaptIntegrator(unittest.TestCase):
    def test_cum_semiinf_adpat_simpson(self):
        def f(x):
            return 10 * np.exp(-x * 3.0) * np.sin(x) + (3 * x) / (2.0 * x + 6.0)

        x_cum, cum, err = xray.cum_int_adapt_simpson(f, 20.0, tol=1e-10)

        ref = 21.834031327325317  # integral from 0 to 20

        i = np.argmin(np.abs(x_cum - 20.0))
        assert x_cum[i] == 20.0
        np.testing.assert_allclose(cum[i], ref, atol=1e-10)
        np.testing.assert_allclose(cum[i], ref, atol=err)

    def test_integral_gauss(self):
        def f(x):
            return np.exp(-(x**2))

        x_cum, cum, err = xray.cum_int_adapt_simpson(f, 10.0, tol=1e-10)

        ref = np.sqrt(np.pi) / 2.0

        np.testing.assert_allclose(cum[-1], ref, atol=1e-10)

    def test_integral_poly(self):
        def f(x):
            return x**3 - x**5

        x_cum, cum, err = xray.cum_int_adapt_simpson(f, 2.0, tol=1e-10)

        ref = 4.0 - 32.0 / 3.0

        np.testing.assert_allclose(cum[-1], ref, atol=1e-10)

    def test_sin(self):
        def f(x):
            return np.cos(x)

        x_cum, cum, err = xray.cum_int_adapt_simpson(f, 250.0, tol=1e-10)
        print(err)
        print(np.max(cum - np.sin(x_cum)))

        np.testing.assert_allclose(cum, np.sin(x_cum), atol=1e-10)

    def test_exp(self):
        def f(x):
            return np.exp(-x)

        x_cum, cum, err = xray.cum_int_adapt_simpson(f, 2000.0, tol=1e-10)
        print(err)
        print(np.max(cum - 1.0 + np.exp(-x_cum)))

        np.testing.assert_allclose(cum, 1.0 - np.exp(-x_cum), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
