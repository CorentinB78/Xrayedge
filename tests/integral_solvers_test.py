import unittest
import numpy as np
from scipy import interpolate
import xrayedge as xray
from xrayedge.integral_solvers import cheb_points

class TestCheb(unittest.TestCase):

    def test_cheb_points(self):
        np.testing.assert_allclose(cheb_points(3), [-1., 0., 1.], atol=1e-15)
        np.testing.assert_allclose(cheb_points(3, -2., 16.), [-2., 7., 16.], atol=1e-15)



class TestSolvePseudoDyson(unittest.TestCase):

    @property
    def solution_ref(self):
        x = np.array([0.        , 0.00308191, 0.01231498, 0.02766126, 0.04905771,
                0.07641638, 0.10962486, 0.1485467 , 0.19302194, 0.24286784,
                0.29787957, 0.35783106, 0.42247597, 0.49154866, 0.5647653 ,
                0.64182501, 0.72241115, 0.80619256, 0.89282499, 0.98195242,
                1.07320862, 1.1662186 , 1.26060016, 1.35596546, 1.45192263,
                1.54807737, 1.64403454, 1.73939984, 1.8337814 , 1.92679138,
                2.01804758, 2.10717501, 2.19380744, 2.27758885, 2.35817499,
                2.4352347 , 2.50845134, 2.57752403, 2.64216894, 2.70212043,
                2.75713216, 2.80697806, 2.8514533 , 2.89037514, 2.92358362,
                2.95094229, 2.97233874, 2.98768502, 2.99691809, 3.        ])

        # computed with Cheb 50 points, accuracy estimated < 1e-4
        y = np.array([-3.56227993+0.j, -3.53858641+0.j, -3.46866995+0.j, -3.35592324+0.j,
                -3.20568068+0.j, -3.02478108+0.j, -2.82103077+0.j, -2.6026321 +0.j,
                -2.37763213+0.j, -2.15345221+0.j, -1.93652033+0.j, -1.73204687+0.j,
                -1.54390907+0.j, -1.37454711+0.j, -1.22576334+0.j, -1.09743128+0.j,
                -0.98923376+0.j, -0.90005678+0.j, -0.82839417+0.j, -0.77249059+0.j,
                -0.73051624+0.j, -0.70067775+0.j, -0.68116526+0.j, -0.67050317+0.j,
                -0.6672488 +0.j, -0.67016811+0.j, -0.67819753+0.j, -0.69042278+0.j,
                -0.70608382+0.j, -0.72446046+0.j, -0.74504178+0.j, -0.76730576+0.j,
                -0.79082084+0.j, -0.81518871+0.j, -0.8400582 +0.j, -0.86508789+0.j,
                -0.88984925+0.j, -0.91443172+0.j, -0.93813583+0.j, -0.96084053+0.j,
                -0.98227983+0.j, -1.0022094 +0.j, -1.02039572+0.j, -1.0366264 +0.j,
                -1.05070782+0.j, -1.06247138+0.j, -1.07177402+0.j, -1.07850205+0.j,
                -1.08257246+0.j, -1.08393491+0.j])

        return interpolate.BarycentricInterpolator(x, y)

    def test_solve_pseudo_dyson_cheb(self):
        V = 2.
        t = 3.

        def cst_func(c):
            return np.vectorize(lambda x: c)

        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(1.), t, V, 10, method='cheb')
        np.testing.assert_allclose(phi, 1. / (1. + V * t))


        time, phi = xray.solve_pseudo_dyson(cst_func(0.), cst_func(1.), t, V, 10, method='cheb')
        np.testing.assert_allclose(phi, 0.)


        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(0.), t, V, 20, method='cheb')
        np.testing.assert_allclose(phi, np.exp(V * (time - t)))


        time, phi = xray.solve_pseudo_dyson(np.sin, np.cos, t, V, 50, method='cheb')
        np.testing.assert_allclose(phi, self.solution_ref(time), atol=1e-4)


    def test_solve_pseudo_dyson_multicheb(self):
        V = 2.
        t = 3.

        def cst_func(c):
            return np.vectorize(lambda x: c)

        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(1.), t, V, 10, method='multicheb')
        np.testing.assert_allclose(phi, 1. / (1. + V * t))


        time, phi = xray.solve_pseudo_dyson(cst_func(0.), cst_func(1.), t, V, 10, method='multicheb')
        np.testing.assert_allclose(phi, 0.)


        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(0.), t, V, 20, method='multicheb')
        np.testing.assert_allclose(phi, np.exp(V * (time - t)))


        time, phi = xray.solve_pseudo_dyson(np.sin, np.cos, t, V, 50, method='multicheb')
        np.testing.assert_allclose(phi, self.solution_ref(time), atol=1e-4)


    def test_solve_pseudo_dyson_trapz(self):
        V = 2.
        t = 3.

        def cst_func(c):
            return np.vectorize(lambda x: c)

        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(1.), t, V, 10, method='trapz')
        np.testing.assert_allclose(phi, 1. / (1. + V * t))


        time, phi = xray.solve_pseudo_dyson(cst_func(0.), cst_func(1.), t, V, 10, method='trapz')
        np.testing.assert_allclose(phi, 0.)


        time, phi = xray.solve_pseudo_dyson(cst_func(1.), cst_func(0.), t, V, 1000, method='trapz')
        np.testing.assert_allclose(phi, np.exp(V * (time - t)), rtol=1e-4, atol=1e-4)


        time, phi = xray.solve_pseudo_dyson(np.sin, np.cos, t, V, 100, method='trapz')
        np.testing.assert_allclose(phi, self.solution_ref(time), atol=1e-3)

class TestCumAdaptIntegrator(unittest.TestCase):

    def test_cum_semiinf_adpat_simpson(self):
        def f(x):
            return 10 * np.exp(-x * 3.) * np.sin(x) + (3 * x) / (2. * x + 6.)

        x_cum, cum, err = xray.cum_semiinf_adpat_simpson(f, 20., tol=1e-10)
        
        ref = 21.834031327325317 # integral from 0 to 20

        i = np.argmin(np.abs(x_cum - 20.))
        assert(x_cum[i] == 20.)
        np.testing.assert_allclose(cum[i], ref, atol=1e-10)
        np.testing.assert_allclose(cum[i], ref, atol=err)

    def test_integral_gauss(self):
        def f(x):
            return np.exp(-x**2)

        x_cum, cum, err = xray.cum_semiinf_adpat_simpson(f, 10., tol=1e-10)
        
        ref = np.sqrt(np.pi) / 2.

        np.testing.assert_allclose(cum[-1], ref, atol=1e-10)

    def test_integral_poly(self):
        def f(x):
            return x**3 - x**5

        x_cum, cum, err = xray.cum_semiinf_adpat_simpson(f, 2., tol=1e-10)
        
        ref = 4. - 32. / 3.

        np.testing.assert_allclose(cum[-1], ref, atol=1e-10)



if __name__ == '__main__':
    unittest.main()