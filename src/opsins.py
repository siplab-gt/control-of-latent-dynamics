"""Dimensionless four-state ChR2 opsin for fSNN's unitless v / Iopto.

Subclass of cleo.opto.opsins.FourStateOpsin (the class returned by cleo's
``chr2_4s()`` factory) that preserves the full nonlinear biophysical model —
Markov kinetics, Hill-function light nonlinearity, singularity-cancelling
fv_timesVminusE voltage dependence — and only changes dimensional plumbing
so it can run on fSNN's ``v : 1`` / ``Iopto : 1`` neuron group.

The three sources of nonlinearity (inherited unchanged from cleo):
  1. Voltage-dependent driving force  — fv_timesVminusE = v1 * (1 - exp(-(v-E)/v0))
  2. Four-state Markov kinetics       — dC1/dt, dO1/dt, dO2/dt with rate coupling
  3. Light saturation (Hill function) — Hp = phi^p / (phi^p + phim^p)

Plumbing changes (vs. the stock volt/ampere-typed FourStateOpsin):
  * ``required_vars`` → ``[("Iopto", radian), ("v", radian)]`` so fSNN's
    dimensionless ng_main passes ``has_same_dimensions`` at injection time.
  * Field defaults for ``g0``, ``E``, ``v0``, ``v1`` are raw ``float``
    (dimensionless). Placeholder defaults are intentionally wrong-scale —
    callers MUST pass calibrated values from ``calibrate_opsin_plant.py``.
  * ``model`` string overridden with unit-annotation-only changes:
    ``: volt`` → ``: 1`` and ``: ampere`` → ``: 1``. The math (including the
    l'Hôpital-transformed fv_timesVminusE form that cancels the v=E
    singularity) is preserved byte-for-byte from cleo. See
    ``scratchpad/v3_opsin_deviations.md`` D4 for rationale.

Kinetic rates (k1, k2, Gf0, kf, Gb0, kb, Gd1, Gd2, Gr0, gamma, phim, p, q)
are inherited unchanged from FourStateOpsin — these are cleo's chr2_4s defaults
derived from Williams et al. 2013 and match what USID's NN opsin was trained
against.
"""
from attrs import asdict, define, field
from brian2.units.allunits import radian
from cleo.opto.opsins import FourStateOpsin


# Bookkeeping fields on Opsin's ancestors (InterfaceDevice → SynapseDevice →
# LightDependent → Opsin → MarkovOpsin) that must NOT end up in the Brian2
# synapse namespace. These are the keys cleo's default `params` implementation
# implicitly excludes by virtue of its "drop everything from __mro__[1]"
# strategy — we enumerate them explicitly below because our override keeps
# parent-class kinetic fields (p, q, k1, …) that cleo would otherwise strip.
_NON_PARAM_FIELDS = frozenset({
    "name", "save_history", "brian_objects", "sim",
    "model", "on_pre", "synapses", "source_ngs",
    "per_ng_unit_replacements", "required_vars",
    "extra_namespace", "spectrum",
})


@define(eq=False)
class UnitlessChr2FourState(FourStateOpsin):
    """chr2_4s (full voltage-dependent nonlinear kinetics) for fSNN's dimensionless v.

    Construction:
        opsin = UnitlessChr2FourState(g0=..., E=..., v0=..., v1=...)

    All four parameters must come from calibrate_opsin_plant.py's output.
    Defaults below are placeholders that will NOT produce fSNN-correct behavior.
    """

    # Dimensionless calibrated parameters — callers MUST override via kwargs.
    g0: float = 1.0
    E: float = 0.0
    v0: float = 1.0
    v1: float = 1.0

    # Accept fSNN's dimensionless v and Iopto. Brian2 checks dimensional
    # compatibility via has_same_dimensions; radian is dimension-empty and
    # matches fSNN's "v : 1" / "Iopto : 1" declarations on ng_main.
    required_vars: list = field(
        factory=lambda: [("Iopto", radian), ("v", radian)],
        init=False,
    )

    # Same math as FourStateOpsin; unit annotations changed from
    # ': volt' → ': 1' and ': ampere' → ': 1'. The fv_timesVminusE
    # l'Hôpital-transformed form is preserved verbatim so the v=E
    # singularity cancels naturally (see cleo's own comment in
    # cleo/opto/opsins.py:119-123).
    model: str = field(
        init=False,
        default="""
        dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1 (clock-driven)
        dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gf)*O1 : 1 (clock-driven)
        dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1 (clock-driven)
        C2 = 1 - C1 - O1 - O2 : 1

        Theta = int(phi_pre > 0*phi_pre) : 1
        Hp = Theta * phi_pre**p/(phi_pre**p + phim**p) : 1
        Ga1 = k1*Hp : hertz
        Ga2 = k2*Hp : hertz
        Hq = Theta * phi_pre**q/(phi_pre**q + phim**q) : 1
        Gf = kf*Hq + Gf0 : hertz
        Gb = kb*Hq + Gb0 : hertz

        fphi = O1 + gamma*O2 : 1
        fv_timesVminusE = v1 * (1 - exp(-(V_VAR_NAME_post - E) / v0)) : 1

        IOPTO_VAR_NAME_post = -g0 * fphi * fv_timesVminusE * rho_rel : 1 (summed)
        rho_rel : 1""",
    )

    # init_syn_vars inherited from FourStateOpsin (sets C1=1, O1=0, O2=0).

    @property
    def params(self) -> dict:
        """All attrs fields go into the Brian2 synapse namespace — including
        those inherited from FourStateOpsin (k1, k2, Gf0, kf, Gb0, kb, Gd1,
        Gd2, Gr0, gamma, phim, p, q).

        Overrides cleo's default `SynapseDevice.params`, which strips every
        field declared on the immediate parent class under the assumption
        that leaf subclasses redeclare everything. Our subclass intentionally
        inherits kinetic rates verbatim — stripping them leaves Brian2 unable
        to resolve `p`, `q`, etc. at equation-parse time.
        """
        params = asdict(self, recurse=False)
        for key in list(params.keys()):
            if key.startswith("_") or key in _NON_PARAM_FIELDS:
                params.pop(key, None)
        return params
