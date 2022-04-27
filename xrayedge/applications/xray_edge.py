import numpy as np
from copy import copy
from ..solver import PhysicsParameters, AccuracyParameters, CorrelatorSolver
from ..reservoir import QPC


class XRayEdgeSolver:
    """
    Solver for computing Green functions of an Anderson impurity capacitively coupled to a QPC.
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

        self.correlator_solver = CorrelatorSolver(QPC(self.PP), self.PP.V_cap, self.AP)

    def weight(self, Q_up, Q_dn):
        return np.exp(
            -self.PP.beta
            * (
                (self.PP.eps_QD - self.PP.mu_QD) * (Q_up + Q_dn)
                + Q_up * Q_dn * self.PP.U
            )
        )

    def Z_d(self):
        """
        Partition function
        """
        return (
            self.weight(0, 0)
            + self.weight(1, 0)
            + self.weight(0, 1)
            + self.weight(1, 1)
        )

    def proba(self, Q_up, Q_dn):
        return self.weight(Q_up, Q_dn) / self.Z_d()

    def G_grea(self, t_array):
        """
        Greater Green function in times on the QD
        """
        prefactor = -1j * np.exp(-1j * t_array * self.PP.eps_QD)
        out = self.proba(0, 0) * self.correlator_solver.A_plus(0, t_array)
        out += (
            np.exp(-1j * t_array * self.PP.U)
            * self.proba(0, 1)
            * self.correlator_solver.A_plus(1, t_array)
        )
        return prefactor * out

    def G_less(self, t_array):
        """
        Lesser Green function in times on the QD
        """
        prefactor = 1j * np.exp(-1j * t_array * self.PP.eps_QD)
        out = self.proba(1, 0) * np.conj(self.correlator_solver.A_minus(1, t_array))
        out += (
            np.exp(-1j * t_array * self.PP.U)
            * self.proba(1, 1)
            * np.conj(self.correlator_solver.A_minus(2, t_array))
        )
        return prefactor * out
