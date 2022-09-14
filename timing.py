"""
Little script to time execution.
"""
import numpy as np
from matplotlib import pyplot as plt
import xrayedge as xray
import toolbox as tb
import time
import cProfile


PP = xray.PhysicsParameters()

tmax = 10.0

AP = xray.AccuracyParameters(
    time_extrapolate=tmax,
    tol_C=1e-3,
    method="trapz-GMRES",
    qdyson_atol=1e-4,
    qdyson_min_step=0.1,
)

qpc = xray.QPC(PP, int(1e4), 100.0)
solver = xray.CorrelatorSolver(qpc, PP.orbitals, PP.couplings, AP)
solver.verbose = True

start = time.time()
start_full = time.process_time()

# cProfile.run("model.compute_C(type=0, Q=0)")
err = solver.compute_C(type=0, Q=0, force_recompute=True)

full_run_time = time.process_time() - start_full
run_time = time.time() - start

print(f"full run time: {full_run_time} s")
print(f"run time: {run_time} s")
print()
print(f"Error: {err}")
print()

times = np.linspace(0.0, tmax, 300)
plt.plot(times, solver.C(0, 0, times).real)
plt.show()
