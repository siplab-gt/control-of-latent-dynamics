"""Dimensionless 4-state ChR2 opsin (Bansal 2020) for fSNN's unitless v / Iopto.

Subclass of cleo.opto.opsins.BansalFourStateOpsin (the Markov 4-state ChR2 model
from Bansal et al. 2020) that preserves the full nonlinear biophysical model —
Markov kinetics, Hill-function light nonlinearity, linear (V−E) driving force —
and only changes dimensional plumbing so it can run on fSNN's
``v : 1`` / ``Iopto : 1`` neuron group.

The two sources of nonlinearity (inherited unchanged from cleo's Bansal):
  1. Four-state Markov kinetics       — dC1/dt, dO1/dt, dO2/dt with rate coupling
  2. Light saturation (Hill function) — Hp = phi^p / (phi^p + phim^p)

The driving force is linear: ``Iopto ∝ (V − E)``. (Bansal differs from PyRhO
chr2_4s here — PyRhO's fv_timesVminusE saturating form has been replaced by
a bare linear term.) The v=E singularity that PyRhO needed l'Hôpital for does
not arise: ``Iopto = 0`` exactly when ``V = E``.

Plumbing changes (vs. the stock volt/ampere-typed BansalFourStateOpsin):
  * ``required_vars`` → ``[("Iopto", radian), ("v", radian)]`` so fSNN's
    dimensionless ng_main passes ``has_same_dimensions`` at injection time.
  * Field defaults for ``g0``, ``E`` are raw ``float`` (dimensionless).
    Placeholder defaults are intentionally wrong-scale — callers MUST pass
    calibrated values from ``calibrate_opsin_plant.py``.
  * ``model`` string overridden with unit-annotation-only changes:
    ``: ampere`` → ``: 1``. The math is preserved byte-for-byte from cleo.
  * Nagel 2003 ChR2 spectrum attached explicitly (cleo's ``chr2_b4s()``
    factory adds this; the bare class does not).

Kinetic rates (k1, k2, Gf0, kf, Gb0, kb, Gd1, Gd2, Gr0, gamma, phim, p, q)
are inherited unchanged from BansalFourStateOpsin — these are cleo's Bansal
defaults from Bansal et al. 2020.
"""
from attrs import asdict, define, field
from brian2.units.allunits import radian
from cleo.opto.opsins import BansalFourStateOpsin


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


# Nagel et al. 2003 ChR2 action spectrum — extracted from cleo's chr2_b4s()
# factory at cleo/opto/opsin_library.py:104-123. The bare BansalFourStateOpsin
# class does not set this; cleo's factory attaches it. We attach it here so
# absorption-at-wavelength behavior matches the factory exactly.
_NAGEL_2003_CHR2_SPECTRUM = [
    (400, 0.34),
    (422, 0.65),
    (460, 0.96),
    (470, 1.0),
    (473, 1.0),
    (500, 0.57),
    (520, 0.22),
    (540, 0.06),
    (560, 0.01),
]


@define(eq=False)
class UnitlessChr2FourState(BansalFourStateOpsin):
    """Bansal 4-state ChR2 (full Markov + Hill nonlinear kinetics) for fSNN's dimensionless v.

    Construction:
        opsin = UnitlessChr2FourState(g0=..., E=...)

    Both parameters must come from calibrate_opsin_plant.py's output.
    Defaults below are placeholders that will NOT produce fSNN-correct behavior.
    """

    # Dimensionless calibrated parameters — callers MUST override via kwargs.
    g0: float = 1.0
    E: float = 0.0

    # Accept fSNN's dimensionless v and Iopto. Brian2 checks dimensional
    # compatibility via has_same_dimensions; radian is dimension-empty and
    # matches fSNN's "v : 1" / "Iopto : 1" declarations on ng_main.
    required_vars: list = field(
        factory=lambda: [("Iopto", radian), ("v", radian)],
        init=False,
    )

    # Nagel 2003 ChR2 action spectrum (peak at 470-473 nm; absorption=1.0 at
    # cleo's default 473 nm fiber). Without this, cleo's LightDependent path
    # uses an undefined-default behavior; attaching it explicitly mirrors what
    # the chr2_b4s() factory does.
    spectrum: list = field(factory=lambda: list(_NAGEL_2003_CHR2_SPECTRUM))

    # Same math as BansalFourStateOpsin; unit annotation changed from
    # ': ampere' → ': 1' on the IOPTO_VAR_NAME line so the summed photocurrent
    # remains dimensionless on fSNN's ng_main.
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

        IOPTO_VAR_NAME_post = -g0*fphi*(V_VAR_NAME_post-E)*rho_rel : 1 (summed)
        rho_rel : 1""",
    )

    # init_syn_vars inherited from BansalFourStateOpsin (sets C1=1, O1=0, O2=0).

    @property
    def params(self) -> dict:
        """All attrs fields go into the Brian2 synapse namespace — including
        those inherited from BansalFourStateOpsin (k1, k2, Gf0, kf, Gb0, kb,
        Gd1, Gd2, Gr0, gamma, phim, p, q).

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
