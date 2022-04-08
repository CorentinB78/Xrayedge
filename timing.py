import numpy as np
from matplotlib import pyplot as plt
import xrayedge as xray
import toolbox as tb
import time


PP = xray.PhysicsParameters()

PP.beta = 50.0
PP.bias = 0.0
PP.eps_c = 0.0  # on the QPC
PP.mu_c = 0.0
PP.Gamma = 1.0
PP.capac_inv = 5.0  # = dV/dQ

tmax = 30.0

AP = xray.AccuracyParameters(
    PP,
    time_extrapolate=tmax,
    tol_C=1e-3,
    delta_interp_phi=0.05,
    fft_w_max=500.0,
    fft_nr_samples=500000,
)

model = xray.NumericModel(PP, AP)

start = time.time()
start_full = time.process_time()

err = model.compute_C(type=0, Q=0)

full_run_time = time.process_time() - start_full
run_time = time.time() - start

print(f"full run time: {full_run_time} s")
print(f"run time: {run_time} s")
print()
print(f"Error: {err}")
print()

times = np.linspace(0.0, tmax, 300)
plt.plot(times, model.C(0, 0, times).real)
plt.show()
