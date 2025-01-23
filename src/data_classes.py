from dataclasses import dataclass
from typing import Self

import brian2.only as b2
import cleo
import dill
import numpy as np
from jaxtyping import Float, Int


class BaseDataClass:
    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename) -> Self:
        with open(filename, "rb") as f:
            return dill.load(f)


@dataclass
class ExptData(BaseDataClass):
    """should be able to use for sys ID and control experiments data"""

    U: Float[np.ndarray, "trials T k"]
    """Control input in mW/mm2"""
    Y: Float[np.ndarray, "trials T m"]
    """Experimental measurements"""
