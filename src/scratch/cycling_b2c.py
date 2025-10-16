import os

import brian2.numpy_ as np
import brian2cuda
import matplotlib.pyplot as plt
import scipy.io
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    TimedArray,
    defaultclock,
    device,
    hertz,
    linked_var,
    ms,
    mV,
    network_operation,
    run,
    second,
    set_device,
)
from matplotlib.ticker import MaxNLocator

set_device("cuda_standalone", build_on_run=False)
# set_device("cuda_standalone")
# set_device("cuda_standalone", compile=True, run=True,
#            debug=True)
# from brian2cuda.device import CUDAStandaloneDevice

plot_flag = False
N = 800  # number of neurons in the spiking network
dt = 1e-3 * second  # integration timestep

data = scipy.io.loadmat("cycling_data.mat")
factors = data["factors"]
emg = data["emg"]

m_in = 1  # number of inputs
m = emg.shape[1]  # number of outputs
P = factors.shape[1]  # number of factors

np.random.seed(3)

utilde_in = -1 + 2 * np.random.rand(m_in, N).T

A = -1 + 2 * np.random.rand(P, N).T
U0, s, Vt = np.linalg.svd(A, full_matrices=False)

# Determine the rank based on the given tolerance
rank = np.sum(s > max(A.shape) * np.finfo(s.dtype).eps * np.max(s))

# Return the orthonormal basis
V = U0[:, :rank]
V[:, [2, 4, 5, 6, 9, 10]] *= -1

g = 4  # feedback gain in spiking network
muf = -0.3  # recurrent inhibition
gs = 0.11  # gain of slow random synapses
gf = 0.13  # gain of fast random synpases

U = g * np.eye(N)  # orthogonal matrix of norm g that maps from space of N to N
u_in = U @ utilde_in  # input weights for the external input to the spiking model
u = U @ V  # mapping from P-space to N-space

Tdemean = 300

vJsbar = np.load("vJsbar.npy")

TRLS = 300

Tinit = 50
DTRLS = 2
tauV = 1e-2 * second  # decay for voltage in spiking net
taus = 1e-1 * second  # decay for slow synapses
tauf = 5e-3 * second  # decay for fast synapses
vth = 1e-16  # spike threshold
vr = -10  # reset voltage

np.random.seed(1)
Jmu = (
    muf * (1 / (tauf / second)) * 1 / N * np.ones((N, N))
)  # mean recurrent matrix in factor approximating network for spiking neurons

J0f_mat = scipy.io.loadmat("./J0f_randn_seed_1.mat")
J0s_mat = scipy.io.loadmat("./J0s_randn_seed_1.mat")

J0f = J0f_mat["J0f"]
J0s = J0s_mat["J0s"]

J0f = np.asarray(J0f)
J0s = np.asarray(J0s)


v_mat = scipy.io.loadmat("v_randn_seed_3.mat")
v_init = v_mat["v"]
v_init = np.asarray(v_init).flatten()

TTrial = 2000
defaultclock.dt = dt

neurons = NeuronGroup(
    N=N,
    model="""
    vJsbar : 1
    u_in : 1 (constant)
    ttrial = t/ms % 2000 : 1 (shared)
    t_trial_num = t/ms // 2000 : 1 (shared)
    fin_bool = ttrial < 50 : boolean
    u_dot_y : 1
    vinf = J0fbars - vJsbar + u_dot_y + J0ss + J0fs + u_in * 2 * int(fin_bool) : 1
    dv/dt = (v - vinf) * (-1 / tauV) : 1
    dJ0ss/dt = -J0ss/taus : 1
    dJ0fs/dt = -J0fs/tauf : 1
    dJ0fbars/dt = -J0fbars/tauf : 1
    dss/dt = -ss/taus : 1
    dsf/dt = -sf/tauf : 1
    sum_vJsbar : 1
    """,
    method="euler",
    threshold="v>=vth",
    reset="""
    v=vr
    ss += 1
    sf += 1
    """,
    name="neurongroup_main",
)
neurons.u_in = np.array(u_in).flatten()
neurons.v = v_init
neurons.J0ss = np.zeros(N)  # J0 slow input current
neurons.J0fs = np.zeros(N)  # J0 fast input current
neurons.J0fbars = np.zeros(N)  # J0 mean current
neurons.sum_vJsbar = np.zeros(N)
neurons.vJsbar = vJsbar.flatten()
neurons.ss = np.zeros(N)
neurons.sf = np.zeros(N)

statemon_main = StateMonitor(
    neurons,
    ["ttrial", "t_trial_num", "ss", "sf", "v"],
    record=True,
    name="statemonitor_main",
)

synapses = Synapses(
    neurons,
    neurons,
    """
                    w1 : 1
                    w2 : 1
                    w3 : 1
                    """,
    on_pre="""
                    J0ss += w1
                    J0fs += w2
                    J0fbars += w3
                    """,
    name="syn_main2main",
)
synapses.connect()

synapses.w1 = J0s.flatten()
synapses.w2 = J0f.flatten()
synapses.w3 = Jmu.flatten()

spikemon = SpikeMonitor(neurons[:20])

ytildetarr = TimedArray(np.array(factors), dt=defaultclock.dt)
ftarr = TimedArray(np.array(emg), dt=defaultclock.dt)

neurons_z = NeuronGroup(
    N=m,
    model="""
    ttrial = t/ms % 2000 : 1 (shared)
    t_trial_num = t/ms // 2000 : 1 (shared)
    f = ftarr(t % (2000 * ms), i) : 1
    z : 1
    z_mon = f * int(t_trial_num < 50) + z * int(t_trial_num >= 50) : 1
    reach_trial_end_bool_z : boolean (shared)
    """,
    namespace={"ftarr": ftarr},
    name="neurongroup_z",
)

neurons_y = NeuronGroup(
    N=P,
    model="""
    ttrial = t/ms % 2000 : 1 (shared)
    t_trial_num = t/ms // 2000 : 1 (shared)
    ytilde = ytildetarr(t % (2000 * ms), i) : 1
    y : 1
    y_mon = ytilde * int(t_trial_num < 50) + y * int(t_trial_num >= 50) : 1
    err = y_mon - ytilde : 1
    reach_trial_end_bool_y : boolean (shared)
    """,
    namespace={"ytildetarr": ytildetarr},
    name="neurongroup_y",
)

neurons_nMSE = NeuronGroup(
    N=1,
    model="""
    nMSE_numerator : 1
    nMSE_denominator : 1
    nMSE_sum : 1
    nMSE : 1
    """,
    name="neurongroup_nMSE",
)
neurons_nMSE.run_regularly(
    """
    nMSE_sum += nMSE_numerator / (nMSE_denominator + 1e-9)
    """,
    dt=defaultclock.dt,
    name="neurongroup_nMSE_run_regularly_dt",
)
neurons_nMSE.run_regularly(
    """
    nMSE = nMSE_sum / 2000
    """,
    dt=defaultclock.dt * 2000,
    name="neurongroup_nMSE_run_regularly_dTTrial",
)
synapses_y2nMSE = Synapses(
    neurons_y,
    neurons_nMSE,
    model="""
    nMSE_numerator_post = err_pre ** 2 : 1 (summed)
    nMSE_denominator_post = ytilde_pre ** 2 : 1 (summed)
    """,
    name="syn_y2nMSE",
)
synapses_y2nMSE.connect()
statemon_nMSE = StateMonitor(neurons_nMSE, "nMSE", record=[0], name="statemonitor_nMSE")

# To check
synapses_u_dot_y = Synapses(
    neurons,
    neurons_y,
    model="""
    u : 1 (constant)
    u_dot_y_pre = u * y_mon : 1 (summed)
    """,
    name="syn_u_dot_y",
)

synapses_u_dot_y.connect()
synapses_u_dot_y.u = np.array(u).flatten()

# RLS

neurons_rand = NeuronGroup(
    N=1,
    model="""
    randSingle = rand() : 1 (constant over dt)
    """,
    threshold="randSingle < 1.0 / DTRLS",
    name="neurongroup_rand",
)
neurons_s = NeuronGroup(
    N,
    model="""
                        ss : 1
                        sf : 1
                        xPUp_tot : 1
                        xPDown_tot : 1
                        kdm1 : 1
                        kLeft : 1
                        kRight : 1
                        rand_trigger : 1
                        """,
    threshold="rand_trigger < 1.0 / DTRLS",
    name="neurongroup_s",
)
synapses_s2rand = Synapses(
    neurons_s,
    neurons_rand,
    model="""
    rand_trigger_pre = randSingle_post : 1 (summed)
    """,
    name="syn_s2rand",
)
synapses_s2rand.connect()
synapses_main2s = Synapses(
    neurons,
    neurons_s,
    model="""
    ss_post = ss_pre : 1 (summed)
    sf_post = sf_pre : 1 (summed)
    """,
    name="syn_main2s",
)
synapses_main2s.connect(condition="i==j")

# connecting neurons_s and itself
synapses_s2s = Synapses(
    neurons_s,
    neurons_s,
    model="""
                        xPUp : 1
                        xPDown : 1

                        xPUp_tot_pre = xPUp : 1 (summed)
                        xPDown_tot_pre = xPDown : 1 (summed)

                        kdm1_pre = ss_post * xPUp_tot_post + sf_post * xPDown_tot_post : 1 (summed)

                        Pw_A : 1
                        Pw_B : 1
                        Pw_C : 1
                        Pw_D : 1
                        """,
    on_pre={
        "pre_step1_xP": """
                            xPUp = Pw_A * ss_post + Pw_B * sf_post
                            xPDown = Pw_C * ss_post + Pw_D * sf_post
                            """,
        "pre_step2_k": """
                            kLeft_pre = xPUp_tot_pre / (1 + kdm1_pre)
                            kRight_pre = xPDown_tot_pre / (1 + kdm1_pre)
                            """,
        "pre_step3_Pw": """
                            Pw_A -= xPUp_tot_pre * kLeft_post
                            Pw_B -= xPUp_tot_pre * kRight_post
                            Pw_C -= xPDown_tot_pre * kLeft_post
                            Pw_D -= xPDown_tot_pre * kRight_post
                            """,
    },
    name="syn_s2s",
)
synapses_s2s.pre_step1_xP.when = "synapses"
synapses_s2s.pre_step1_xP.order = -5

synapses_s2s.summed_updaters["xPUp_tot_pre"].when = "synapses"
synapses_s2s.summed_updaters["xPUp_tot_pre"].order = synapses_s2s.pre_step1_xP.order + 1
synapses_s2s.summed_updaters["xPDown_tot_pre"].when = "synapses"
synapses_s2s.summed_updaters["xPDown_tot_pre"].order = (
    synapses_s2s.pre_step1_xP.order + 1
)

synapses_s2s.summed_updaters["kdm1_pre"].when = "synapses"
synapses_s2s.summed_updaters["kdm1_pre"].order = (
    synapses_s2s.summed_updaters["xPUp_tot_pre"].order + 1
)

synapses_s2s.pre_step2_k.when = "synapses"
synapses_s2s.pre_step2_k.order = synapses_s2s.summed_updaters["kdm1_pre"].order + 1

synapses_s2s.pre_step3_Pw.when = "synapses"
synapses_s2s.pre_step3_Pw.order = synapses_s2s.pre_step2_k.order + 1

synapses_s2s.connect()
synapses_s2s.Pw_A = 1.0 * np.eye(N).flatten()  # inverse covariance matrix
synapses_s2s.Pw_B = 1.0 * np.eye(N).flatten()  # inverse covariance matrix
synapses_s2s.Pw_C = 1.0 * np.eye(N).flatten()  # inverse covariance matrix
synapses_s2s.Pw_D = 1.0 * np.eye(N).flatten()  # inverse covariance matrix

# connecting neurons_s and neurons_y
synapses_y2s = Synapses(
    neurons_y,
    neurons_s,
    model="""
    wLeft : 1
    wRight : 1
    y_pre = wLeft * ss_post + wRight * sf_post : 1 (summed)
    """,
    on_post="""
    wLeft -= (y_pre - ytilde_pre) * kLeft_post
    wRight -= (y_pre - ytilde_pre) * kRight_post
    """,
    name="syn_y2s",
)
synapses_y2s.connect()
synapses_y2s.wLeft = np.zeros((P, N)).flatten()
synapses_y2s.wRight = np.zeros((P, N)).flatten()

neurons_yPlus = NeuronGroup(
    N=P + 1,
    model="""
    y : 1
    yPlus = y * int(i<P) + 1.0 * int(i>=P) : 1
    xP_tot : 1
    kdm1 : 1
    k : 1
    rand_trigger : 1
    """,
    threshold="rand_trigger < 1.0 / DTRLS",
    name="neurongroup_yPlus",
)
synapses_yPlus2rand = Synapses(
    neurons_yPlus,
    neurons_rand,
    model="""
    rand_trigger_pre = randSingle_post : 1 (summed)
    """,
    name="syn_yPlus2rand",
)
synapses_yPlus2rand.connect()
synapses_y2yPlus = Synapses(
    neurons_y,
    neurons_yPlus,
    model="""
    y_post = y_pre : 1 (summed)
    """,
    name="syn_y2yPlus",
)
synapses_y2yPlus.connect(condition="i == j", skip_if_invalid=True)

# connecting yPlus and itself
synapses_yPlus2yPlus = Synapses(
    neurons_yPlus,
    neurons_yPlus,
    model="""
    xP : 1
    xP_tot_pre = xP : 1 (summed)
    kdm1_pre = yPlus_post * xP_tot_post : 1 (summed)
    PW : 1
    """,
    on_pre={
        "pre_step1_xP": "xP = PW * yPlus_post",
        "pre_step2_k": "k_pre = xP_tot_pre / (1 + kdm1_pre)",
        "pre_step3_PW": "PW -= xP_tot_pre * k_post",
    },
    name="syn_yPlus2yPlus",
)
synapses_yPlus2yPlus.connect()
synapses_yPlus2yPlus.PW = 1.0 * np.eye(P + 1).flatten()

synapses_yPlus2yPlus.pre_step1_xP.when = "synapses"
synapses_yPlus2yPlus.pre_step1_xP.order = -5

synapses_yPlus2yPlus.summed_updaters["xP_tot_pre"].when = "synapses"
synapses_yPlus2yPlus.summed_updaters["xP_tot_pre"].order = (
    synapses_yPlus2yPlus.pre_step1_xP.order + 1
)

synapses_yPlus2yPlus.summed_updaters["kdm1_pre"].when = "synapses"
synapses_yPlus2yPlus.summed_updaters["kdm1_pre"].order = (
    synapses_yPlus2yPlus.summed_updaters["xP_tot_pre"].order + 1
)

synapses_yPlus2yPlus.pre_step2_k.when = "synapses"
synapses_yPlus2yPlus.pre_step2_k.order = (
    synapses_yPlus2yPlus.summed_updaters["kdm1_pre"].order + 1
)

synapses_yPlus2yPlus.pre_step3_PW.when = "synapses"
synapses_yPlus2yPlus.pre_step3_PW.order = synapses_yPlus2yPlus.pre_step2_k.order + 1

# connecting yPlus and z
synapses_z2yPlus = Synapses(
    neurons_z,
    neurons_yPlus,
    model="""
    W : 1
    z_pre = W * yPlus_post : 1 (summed)
    """,
    on_post="""
    W -= (z_pre - f_pre) * k_post
    """,
    name="syn_z2yPlus",
)
synapses_z2yPlus.connect()
synapses_z2yPlus.W = np.zeros((m, P + 1)).flatten()


statemon_ys = StateMonitor(
    neurons_y,
    ["y_mon", "ytilde", "ttrial", "t_trial_num"],
    record=True,
    name="statemon_ys",
)
statemon_zs = StateMonitor(neurons_z, ["z_mon", "f"], record=True, name="statemon_zs")


neurons_rand.active = False
neurons_s.active = False
synapses_s2rand.active = False
synapses_main2s.active = False
synapses_s2s.active = False
synapses_y2s.active = False
neurons_yPlus.active = False
synapses_yPlus2rand.active = False
synapses_y2yPlus.active = False
synapses_yPlus2yPlus.active = False
synapses_z2yPlus.active = False
spikemon.active = False
statemon_ys.active = False
statemon_zs.active = False

run(TTrial * dt * Tinit, report="text", profile=True)


neurons_rand.active = True
neurons_s.active = True
synapses_s2rand.active = True
synapses_main2s.active = True
synapses_s2s.active = True
synapses_y2s.active = True
neurons_yPlus.active = True
synapses_yPlus2rand.active = True
synapses_y2yPlus.active = True
synapses_yPlus2yPlus.active = True
synapses_z2yPlus.active = True

T = 300
run(TTrial * dt * (T - 1), report="text", profile=True)

spikemon.active = True
statemon_ys.active = True
statemon_zs.active = True

run(TTrial * dt * 1, report="text", profile=True)
device.build(clean=True, compile=True, run=True, debug=False)
# device.build()
# print(scheduling_summary())

output_dir = "output_fig"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(30, 30))
axes = axes.flatten()

plot_flag = True

if plot_flag:
    axes[0].set_title("Spike raster")
    axes[0].scatter(
        spikemon.t / second,
        spikemon.i,
        marker="|",
        linestyle="None",
        color="black",
        s=100,
    )
    axes[0].set_ylim(0, len(spikemon.source))
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("Neuron")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].set_title("Output")
    for output_index in range(m):
        axes[1].plot(statemon_zs.t / second, statemon_zs.z_mon[output_index])
        axes[1].plot(
            statemon_zs.t / second, statemon_zs.f[output_index], linestyle="--"
        )

    axes[2].set_title("Factors")
    for factor_index in range(P):
        print(f"statemon_ys.y_mon[{factor_index}]: {statemon_ys.y_mon[factor_index]}")
        axes[2].plot(statemon_ys.t / second, statemon_ys.y_mon[factor_index])
        axes[2].plot(
            statemon_ys.t / second, statemon_ys.ytilde[factor_index], linestyle="--"
        )
    axes[2].set_xlabel("t [s]")

output_path = os.path.join(output_dir, "output_figure.png")
fig.tight_layout()

# Save the figure
fig.savefig(output_path, dpi=300)

# Optionally print out the location of the saved file
print(f"Figure saved to {output_path}")
