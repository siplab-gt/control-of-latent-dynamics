from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass


@dataclass
class LDS(BaseDataClass):
    """Linear Dynamical System model.

    Note: The following attributes are added dynamically by model_fit methods
    (usid.py, braid.py, known_factors.py) and are not declared dataclass fields:
    - Cz: behavioral decoder matrix (required for m_z property and x2z method)
    - x2z: method to map states to behavioral output
    - y_mean, y_std, u_mean, u_std, x_mean, x_std, z_mean, z_std: normalization stats
    """
    A: Float[np.ndarray, "{self.n} {self.n}"]
    B: Float[np.ndarray, "{self.n} {self.k}"]
    C: Float[np.ndarray, "{self.m} {self.n}"]
    Q: Float[np.ndarray, "{self.n} {self.n}"]
    R: Float[np.ndarray, "{self.m} {self.m}"]
    k_opto: int
    linreg_lds2fsnn: LinearRegression = None
    """to map from model X to ground-truth fSNN X"""

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.C.shape[0]

    @property
    def m_z(self) -> int:
        # Note: Cz is added dynamically by model_fit methods, not a declared field
        return self.Cz.shape[0]

    @property
    def k(self) -> int:
        return self.B.shape[1]

    @property
    def k_task(self) -> int:
        return self.k - self.k_opto

    @property
    def B_opto(self) -> Float[np.ndarray, "{self.n} {self.k_opto}"]:
        return self.B[:, : self.k_opto]

    @property
    def B_task(self) -> Float[np.ndarray, "{self.n} {self.k_task}"]:
        return self.B[:, self.k_opto :]


