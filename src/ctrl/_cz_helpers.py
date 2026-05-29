"""Shared Cz dispatch helpers for the PyTorch-based controllers (NLMPC, iLQR).

The LDS abstraction exposes Cz in one of three forms:

  1. **Linear unified-LDS** (BRAID linear_Cz=True, USID after composition):
     `lds.Cz` is a `(m_z, n)` numpy array.
  2. **Linear subspace-LDS** (USID before composition):
     `lds.Cz1`, `lds.Cz2` are `(m_z, n1)`, `(m_z, n2)`; concatenate as
     `[Cz1, Cz2]` to recover the full Cz.
  3. **Nonlinear torch-Cz** (BRAID linear_Cz=False):
     `lds.cz_torch_module` is a `TorchNonlinearCz` (`nn.Module`) and
     `lds.Cz` is absent.

Both NLMPC and iLQR need to:

  * Apply Cz to a state during forward rollouts (`apply_cz_torch`).
  * (iLQR only) Take the Jacobian of Cz at a state for the quadratic
    cost expansion in the LQR backward pass (`cz_jacobian`).

This module collapses those cases into one place so the controllers
don't each re-implement the three-way dispatch.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def build_cz_state(lds) -> Dict[str, object]:
    """Detect linear-vs-nonlinear Cz from an LDS and return a controller-
    friendly state dict.

    Returns one of:
        {'is_nonlinear': True,  'cz_module': nn.Module}
        {'is_nonlinear': False, 'Cz': torch.Tensor (m_z, n)}
        {'is_nonlinear': False}  # no Cz (ref_type='X' path; passthrough)

    The dict carries the explicit `is_nonlinear` flag rather than relying
    on key presence so callers can use a single branch.
    """
    cz_module = getattr(lds, 'cz_torch_module', None)
    if cz_module is not None:
        if not isinstance(cz_module, nn.Module):
            raise TypeError(
                f"lds.cz_torch_module must be an nn.Module, got "
                f"{type(cz_module).__name__}"
            )
        cz_module.eval()
        for p in cz_module.parameters():
            p.requires_grad_(False)
        return {'is_nonlinear': True, 'cz_module': cz_module}

    cz_matrix = getattr(lds, 'Cz', None)
    if cz_matrix is not None:
        return {
            'is_nonlinear': False,
            'Cz': torch.tensor(cz_matrix, dtype=torch.float32),
        }

    cz1 = getattr(lds, 'Cz1', None)
    if cz1 is not None:
        Cz2 = getattr(lds, 'Cz2', None)
        if Cz2 is None:
            raise AttributeError("lds.Cz1 present but lds.Cz2 missing")
        Cz = np.hstack([cz1, Cz2])
        return {
            'is_nonlinear': False,
            'Cz': torch.tensor(Cz, dtype=torch.float32),
        }

    return {'is_nonlinear': False}  # ref_type='X' fallback


def apply_cz(cz_state: Dict[str, object],
             x: torch.Tensor) -> torch.Tensor:
    """Map state(s) through Cz: `(..., n) → (..., m_z)`.

    Linear case: matrix multiply against `cz_state['Cz']` (broadcasts over
    leading batch dims). Nonlinear case: call `cz_state['cz_module']`
    directly (nn.Module forward broadcasts over leading batch dims too).
    If no Cz is configured, returns `x` unchanged (ref_type='X' path).
    """
    if cz_state.get('is_nonlinear'):
        return cz_state['cz_module'](x)
    Cz = cz_state.get('Cz')
    if Cz is None:
        return x
    return x @ Cz.T


def cz_jacobian(cz_state: Dict[str, object],
                x: torch.Tensor) -> torch.Tensor:
    """Jacobian `∂(Cz·x)/∂x` evaluated at a single state `x` of shape (n,).

    Returns `(m_z, n)` tensor.

    For linear Cz this is constant (returns `cz_state['Cz']` itself). For
    nonlinear Cz it's computed via `torch.autograd.functional.jacobian`
    at the operating point — used by iLQR to build the Gauss-Newton
    quadratic-cost expansion in the backward pass.
    """
    if cz_state.get('is_nonlinear'):
        # x must be 1-D; jacobian computes ∂forward/∂x at that point.
        if x.dim() != 1:
            raise ValueError(
                f"cz_jacobian expects a 1-D state vector, got shape {tuple(x.shape)}"
            )
        return torch.autograd.functional.jacobian(
            cz_state['cz_module'], x, vectorize=True,
        )
    Cz = cz_state.get('Cz')
    if Cz is None:
        raise ValueError("cz_jacobian called but no Cz is configured on this LDS")
    return Cz


def has_cz(cz_state: Dict[str, object]) -> bool:
    """Whether the LDS has any Cz (matrix or nonlinear). Used to decide
    ref_type='Z' vs passthrough ref_type='X' fallback in rollouts."""
    return cz_state.get('is_nonlinear') or ('Cz' in cz_state)
