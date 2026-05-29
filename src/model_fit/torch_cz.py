"""Torch port of BRAID's nonlinear Cz (RegressionModel MLP).

BRAID trains its Cz mapping (latent state → behavioral output Z) as a small
TensorFlow MLP under the `Cz1HL64U` method-code convention (1 hidden layer,
64 units, ReLU). The downstream MPC controllers operate on PyTorch tensors
(for NLMPC/iLQR autograd) or build CVXPY problems (for LQMPC). Neither can
take the TF model directly.

This module:
  * Defines a small `torch.nn.Module` mirroring the BRAID Cz topology.
  * Provides `from_braid_regression()` to copy weights from a trained
    BRAID `RegressionModel` into the torch module (bit-exact transfer
    verified by a scratch port test, not committed).

Stored on the `LDS` as `lds.cz_torch_module` after model_fit. The ctrl-side
helper `src/ctrl/_cz_helpers.py` dispatches on this attribute to decide
linear-vs-nonlinear handling in the rollouts.

Only the architecture used by USID/COLD (`Cz1HL64U` and minor scaling
variants — 1+ hidden layers, ReLU activation, linear output) is supported.
Other BRAID variants (poisson out_dist, classification, dropout) would
require extending this module.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class TorchNonlinearCz(nn.Module):
    """Torch nn.Module mirroring BRAID's `Cz<N>HL<U>U` MLP.

    Architecture (for num_hidden_layers=N, hidden_units=U):
        Dense(n_in, U) + ReLU
        Dense(U, U) + ReLU       (repeated N-1 times for N ≥ 2)
        Dense(U, n_out)          [linear output]
    """

    def __init__(self, n_in: int, n_out: int, hidden_units: int = 64,
                 num_hidden_layers: int = 1, activation: str = 'relu'):
        super().__init__()
        if activation != 'relu':
            raise NotImplementedError(
                f"Only relu activation supported; got {activation!r}. "
                f"Extend if needed."
            )
        layers = []
        prev = n_in
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_units))
            layers.append(nn.ReLU())
            prev = hidden_units
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.hidden_units = int(hidden_units)
        self.num_hidden_layers = int(num_hidden_layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Cz to a state tensor.

        Accepts any shape `(..., n_in)` and returns `(..., n_out)`. Used
        both in batched rollouts (shape `(H, n_in)`) and single-step
        Jacobian computation (shape `(n_in,)`).
        """
        return self.net(x)

    @classmethod
    def from_braid_regression(cls, braid_reg_model, n_in: int, n_out: int,
                              hidden_units: int = 64,
                              num_hidden_layers: int = 1,
                              activation: str = 'relu') -> 'TorchNonlinearCz':
        """Copy weights from a trained BRAID `RegressionModel` into a torch
        module. Bit-exact transfer (see test_torch_cz_port.py).

        BRAID's `.layers` contains `tf.keras.layers.Dense` instances
        (plus optional Dropout, which we skip). Each Dense layer has
        Keras-convention weights `kernel` of shape `(in_dim, out_dim)`;
        torch.nn.Linear's `.weight` is `(out_dim, in_dim)` — transpose to
        bridge.
        """
        import tensorflow as tf  # local: braid.py already imports TF
        model = cls(n_in=n_in, n_out=n_out, hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers, activation=activation)
        dense_layers = [l for l in braid_reg_model.layers
                        if isinstance(l, tf.keras.layers.Dense)]
        torch_linears = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
        if len(dense_layers) != len(torch_linears):
            raise ValueError(
                f"Layer count mismatch: BRAID has {len(dense_layers)} Dense, "
                f"torch has {len(torch_linears)} Linear. "
                f"BRAID layer types: {[type(l).__name__ for l in braid_reg_model.layers]}"
            )
        for braid_layer, torch_layer in zip(dense_layers, torch_linears):
            W = braid_layer.kernel.numpy()
            b = braid_layer.bias.numpy()
            if torch_layer.weight.shape != (W.shape[1], W.shape[0]):
                raise ValueError(
                    f"Shape mismatch: torch {tuple(torch_layer.weight.shape)} "
                    f"vs Keras (in,out)={W.shape}"
                )
            torch_layer.weight.data = torch.tensor(W.T, dtype=torch.float32)
            torch_layer.bias.data = torch.tensor(b, dtype=torch.float32)
        log.info(
            f"TorchNonlinearCz: ported {len(dense_layers)} Dense layers from "
            f"BRAID (n_in={n_in}, n_out={n_out}, hidden_units={hidden_units}, "
            f"num_hidden_layers={num_hidden_layers})"
        )
        return model


def torch_cz_predict(cz_module: TorchNonlinearCz,
                     X: np.ndarray) -> np.ndarray:
    """Convenience wrapper: numpy-in, numpy-out, batch-first convention.

    X: shape `(n_in, T)` (BRAID convention) → returns `(n_out, T)`.

    Used by braid.py to sanity-check that the torch port matches BRAID's
    predictions on training data after weight transfer.
    """
    cz_module.eval()
    with torch.no_grad():
        x_t = torch.tensor(X.T, dtype=torch.float32)  # (T, n_in)
        y_t = cz_module(x_t).numpy()                  # (T, n_out)
    return y_t.T
