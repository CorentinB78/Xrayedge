import numpy as np
from copy import copy
from ..solver import PhysicsParameters, AccuracyParameters, CorrelatorSolver


class NCASolver:
    """
    Solver for computing Pseudo-particle Green functions of an Anderson impurity capacitively coupled to a QPC.
    """

    def __init__(self, physics_params=None, accuracy_params=None):
        self.PP = (
            copy(physics_params) if physics_params is not None else PhysicsParameters()
        )
        self.AP = (
            copy(accuracy_params)
            if accuracy_params is not None
            else AccuracyParameters(self.PP, 1.0)
        )

        self.correltor_solver = CorrelatorSolver(self.PP, self.AP)

    def G_grea_NCA_constraint(self, t_array):
        """
        Greater Green function in times on the QD under NCA constrain.
        """
        # no U in NCA constraint
        return (
            -1j
            * np.exp(-1j * t_array * self.PP.eps_d)
            * self.weight(0, 0)
            * self.correltor_solver.A_plus(0, t_array)
        )

    def G_reta_w_NCA_constraint(self, nr_freqs):
        """
        Greater-retarded Green function in frequencies on the QD under NCA constrain.

        For NCA in the steady state regime, one only needs the greater quaisparticle GFs in the sector Q=0 (see notes).
        Also, the partition function is reduced to 1.

        Returns: freqs, G_grea, energy shift
        """
        # no U in NCA constraint
        w, A_w, energy_shift = self.correltor_solver.A_plus_reta_w(0, nr_freqs)
        return w, -1j * self.weight(0, 0) * A_w, energy_shift - self.PP.eps_d
