from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass


@dataclass
class LDS(BaseDataClass):
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
