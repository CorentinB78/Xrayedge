import numpy as np
from matplotlib import pyplot as plt
import xrayedge as xray
import toolbox as tb
import time


PP = xray.PhysicsParameters()

PP.beta = 50.
PP.bias = 0.
PP.eps_c = 0. # on the QPC
PP.mu_c = 0.
PP.Gamma = 1.
PP.capac_inv = 5. # = dV/dQ

times = np.linspace(0, 50, 5000)

AP = xray.AccuracyParameters(PP,
                            tol_C=1e-3,
                            delta_interp_phi=0.05,
                            fft_w_max=500.,
                            fft_nr_samples=500000,
                            )

model = xray.NumericModel(PP, AP)

start = time.time()
start_full = time.process_time()

tt, C, err = model.C(30., 0)

full_run_time = time.process_time() - start_full
run_time = time.time() - start

print(f"full run time: {full_run_time} s")
print(f"run time: {run_time} s")
print()
print(f"Error: {err}")
print()

plt.plot(tt, C.real)
plt.show()