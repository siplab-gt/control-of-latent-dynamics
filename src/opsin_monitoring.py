"""Per-neuron opsin activation recording, shared by sys_id_expts and ctrl_expts.

Attaches Brian2 StateMonitors on ng_main to record the per-neuron photocurrent
Iopto (always) and membrane voltage v (optional), and computes the static
channel->neuron transmittance. Post-run, subsamples both onto the (T, N) grid
that X/Y/Z use. One source of truth for the recording so sys-ID and control
emit identically-formatted per-neuron activation data.
"""
import brian2.only as b2
import cleo
import numpy as np


def attach_opsin_monitors(cleo_prep, record_v=False, record_dt=None):
    """Attach a per-neuron Iopto StateMonitor (and a v monitor if record_v) on
    ng_main, and compute the static channel->neuron transmittance. Call BEFORE
    run_trials; extract the recordings afterward with extract_per_neuron.

    cleo opsin classes declare Iopto on ng_main via a (summed) synapse
    expression, so monitoring ng_main directly gives the final per-neuron
    photocurrent regardless of opsin class. The transmittance T[c,n] = sum of
    per-source transmittances for sources in channel c; multiplying by Uopto(t)
    reconstructs per-neuron irradiance.

    record_dt (a Brian2 time Quantity) sets the StateMonitor sampling interval.
    None records every network step (sys-ID's behavior; subsample afterward in
    extract_per_neuron). Passing the subsampled period (subsample_factor *
    fsnn.dt) records directly on the X/Y/Z grid -- ~subsample_factor× less RAM,
    which matters for the full-N control run (recording all neurons at the 1 ms
    network step would be multiple GB and OOM the ctrl_expts cgroup).

    Returns (monitors, channel_neuron_transmittance) where monitors is a dict
    with key "iopto" and, if record_v, "v".
    """
    ng_main = cleo_prep.fsnn.ng_main
    mon_kw = {} if record_dt is None else {"dt": record_dt}
    monitors = {
        "iopto": b2.StateMonitor(
            ng_main, "Iopto", record=True, name="opsin_iopto_mon", **mon_kw
        )
    }
    cleo_prep.fsnn.net.add(monitors["iopto"])
    if record_v:
        monitors["v"] = b2.StateMonitor(
            ng_main, "v", record=True, name="opsin_v_mon", **mon_kw
        )
        cleo_prep.fsnn.net.add(monitors["v"])

    neuron_coords = cleo.coords.coords_from_ng(ng_main)
    trans_per_source = cleo_prep.light.transmittance(neuron_coords)  # (n_sources, N)
    k_channels = cleo_prep.light.n_groups
    N = ng_main.N
    channel_neuron_transmittance = np.zeros((k_channels, N), dtype=np.float32)
    for c, group in enumerate(cleo_prep.light.groups):
        channel_neuron_transmittance[c] = trans_per_source[group].sum(axis=0)
    return monitors, channel_neuron_transmittance


def extract_per_neuron(monitors, subsample_factor):
    """Post-run: subsample recorded Iopto (and v if present) onto the (T, N)
    grid. StateMonitor shape is (N_neurons, T_brian); transpose + subsample ->
    (T, N). Returns (Iopto_per_neuron, V_per_neuron); V_per_neuron is None when
    v was not recorded.

    Iopto can be:
      * ampere-typed (ProportionalCurrentOpsin on fSNN) -- convert to nA to
        match the scale USID's NN opsin was trained against
        (synthetic_opsin_data.py emits nA for amp-typed Iopto).
      * dimensionless (UnitlessChr2FourState on fSNN's `Iopto : 1`) -- pass
        through unchanged; dividing by b2.namp would be a DimensionMismatchError.
    Dimension-pattern-match instead of isinstance() so an empty-dim Quantity
    doesn't silently trip the amp branch.

    NOTE to future maintainers: the check uses `b2.amp` (broad ampere family)
    while the divisor is `b2.namp` (nanoampere). These look inconsistent but
    both are amp-dimension units; `have_same_dimensions` checks dimensions not
    scale. If you change the divisor to `b2.amp` to "match the check," you'll
    silently shift the stored scale by 1e9 -- downstream USID diagnostics expect
    the nA scale because synthetic_opsin_data.py emits nA.
    """
    iopto_raw = monitors["iopto"].Iopto
    if b2.have_same_dimensions(iopto_raw, b2.amp):
        iopto_raw = iopto_raw / b2.namp
    Iopto_per_neuron = np.asarray(iopto_raw).T[::subsample_factor].astype(np.float32)
    V_per_neuron = None
    if "v" in monitors:
        V_per_neuron = np.asarray(monitors["v"].v).T[::subsample_factor].astype(np.float32)
    return Iopto_per_neuron, V_per_neuron
