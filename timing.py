"""
Little script to time execution.
"""
import numpy as np
from matplotlib import pyplot as plt
import xrayedge as xray
import time
import cProfile


PP = xray.PhysicsParameters()
PP.eps_res = [0.0, 0.0, 0.0]
PP.orbitals = [0, 1, 2]
PP.couplings = [1.0, 0.5, -0.7]

tmax = 10.0

AP = xray.AccuracyParameters(
    time_extrapolate=tmax,
    tol_C=1e-3,
    method="trapz-GMRES",
    qdyson_atol=1e-4,
    qdyson_min_step=0.1,
    parallelize_orbitals=True,
)

qpc = xray.ExtendedQPC(PP, int(1e4), 100.0)
qpc.verbose = True
solver = xray.CorrelatorSolver(qpc, PP.orbitals, PP.couplings, AP)
solver.verbose = True

start = time.time()
# start_full = time.process_time()

# cProfile.run("solver.compute_C(type=0, Q=0, force_recompute=True)")
err = solver.compute_C(type=0, Q=0, force_recompute=True)

# full_run_time = time.process_time() - start_full
run_time = time.time() - start

# print(f"full run time: {full_run_time} s")
print(f"run time: {run_time} s")
print()
print(f"Error: {err}")
print()

times = np.linspace(0.0, 1.5 * tmax, 1000)
plt.plot(times, solver.C(0, 0, times).real)
plt.show()

solver.plot_nr_GMRES_iter()
