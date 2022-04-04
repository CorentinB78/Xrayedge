import numpy as np
from scipy import integrate


beta = 50.
# mu = 0. # chemical potential on the QD
bias = 0.
# capac_inv = 1. # = dV/dQ
# eps_d = 0. # on the QD
# eps_c = 0. # on the QPC
# mu_QPC = 0.
# Gamma = 2.

nr_channels = 4
lambda_phi = 1. # = capac ?
lambda_chi = 0. # = capac ?
ksi_0 = 10.
    

# def Fermi_dos(self):
#     return Gamma / (eps_d**2 + Gamma**2) / np.pi
    

def A_plus(t_array, beta, bias, lambda_phi, lambda_chi, ksi_0):
    nr_channels = 4
    fermi_dos = 1.
    delta_phi = np.arctan(np.pi * lambda_phi * fermi_dos)

    alpha = nr_channels * (delta_phi / np.pi) ** 2
    gamma = nr_channels * (lambda_chi * fermi_dos) ** 2 * np.cos(delta_phi) ** 4
    # print(alpha, gamma)

    def h_integrand(u):
        return np.divide(u * (1. - np.cos(bias * u)) * np.pi**2, 
                            (beta * np.sinh(np.pi * u / beta))**2, 
                            where=u != 0.)

    assert(t_array[0] == 0.0)

    if bias != 0.0:
        h = integrate.cumulative_trapezoid(x=t_array, 
                                        y=h_integrand(t_array), 
                                        initial=0.)
    else:
        h = 0.

    # TODO: improve integration

    left_chunk = np.divide(1j * np.pi * np.ones_like(t_array), 
                           (beta * ksi_0 * np.sinh(np.pi * t_array / beta)), 
                           where=t_array != 0.)
    left_chunk[t_array == 0.] = np.nan
    right_chunk = np.exp(-np.pi * gamma * np.abs(bias) * t_array + gamma * h)

    return pow(left_chunk, alpha + gamma) * right_chunk
    