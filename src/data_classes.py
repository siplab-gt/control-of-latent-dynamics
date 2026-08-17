from dataclasses import dataclass
from typing import Self, Optional

import numpy as np

from src.helpers import load_pkl_gz, save_pkl_gz


class BaseDataClass:
    def save(self, filename):
        save_pkl_gz(self, filename)

    @classmethod
    def load(cls, filename) -> Self:
        return load_pkl_gz(filename)


@dataclass
class COLDUSIDHandoff(BaseDataClass):
    """
    Handoff data structure for COLD-USID integration.

    Contains raw data from COLD simulations. Normalization is handled
    by model_fit.py (single source of truth for normalization stats).

    USID training uses two-stage approach:
    1. Natural model trained on natural_Y (no opto, pure intrinsic dynamics)
    2. Full USID model trained on opto_Y/opto_U (with opto stimulation)
    """
    # Natural data (from fsnn_test_data - no opto)
    natural_Y: np.ndarray   # (T_nat, m) observations during natural behavior
    natural_Z: np.ndarray   # (T_nat, m_z) behavioral output during natural behavior

    # Opto-stimulated data (from sys_id_data)
    opto_Y: np.ndarray      # (T_opto, m) observations with opto
    opto_Z: np.ndarray      # (T_opto, m_z) behavioral output with opto
    opto_U: np.ndarray      # (T_opto, k) optogenetic inputs
    X_fsnn: np.ndarray      # (T_opto, n) ground truth fSNN latent factors

    # Metadata
    n_x: int                # State dimension (fSNN latent factors)
    k_opto: int             # Number of optogenetic input channels
    seed: int               # Random seed for reproducibility
    wildcards_hash: str     # Hash of COLD wildcards for deterministic output naming

    # Data masks/tags for evaluation
    spiked_mask: Optional[np.ndarray] = None  # (T_opto,) boolean mask where stimulation is active

    @property
    def m(self) -> int:
        """Observation dimension."""
        return self.opto_Y.shape[1]

    @property
    def m_z(self) -> int:
        """Behavioral output dimension."""
        return self.opto_Z.shape[1]

    @property
    def k(self) -> int:
        """Input dimension (optogenetic channels)."""
        return self.opto_U.shape[1]

    @property
    def T_natural(self) -> int:
        """Length of natural data."""
        return self.natural_Y.shape[0]

    @property
    def T_opto(self) -> int:
        """Length of opto data."""
        return self.opto_Y.shape[0]
