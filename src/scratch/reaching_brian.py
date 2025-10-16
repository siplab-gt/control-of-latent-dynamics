import brian2.numpy_ as np
import matplotlib.pyplot as plt
import scipy.io
from brian2 import (
    Hz,
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    defaultclock,
    ms,
    network_operation,
    run,
    second,
)

# def assert_shape(var, expected_shape=None, var_name="variable"):
#     if expected_shape is None:
#         # Check if the variable is a scalar
#         assert np.isscalar(var), f"{var_name} is {type(var)}, expected a scalar"
#     else:
#         # Check if the variable has the expected shape
#         assert isinstance(var, np.ndarray), f"{var_name} is {type(var)}, expected np.ndarray"
#         assert var.shape == expected_shape, f"{var_name} has shape {var.shape}, expected {expected_shape}"


def trial_cycling(dt):
    # timing of event sequence, for each trial
    event = [0.05, 1.95]
    event = [int(e / dt) for e in event]

    TTrial = sum(event)  # length of a trial
    fin = np.concatenate(
        [2.0 * np.ones(event[0]), np.zeros(sum(event[1:]))]
    )  # concatenate various elements into fin and fout
    return fin, TTrial


def get_TTrial(dt):
    event = [0.05, 1.95]
    return np.sum(event) * second / dt


class SpikingNetwork:
    def __init__(self):
        self.N = None  # number of neurons in the spiking network
        self.Ntilde = None
        self.u = None
        self.u_in = None
        self.utilde = None
        self.utilde_in = None
        self.Jtilde = None
        self.muf = None
        self.gs = None
        self.gf = None
        self.dt = None  # integration timestep

    def spiking_network(
        self,
        task="reaching",
        mode=None,
        T=None,
        V=None,
        emg=None,
        vJsbar_input=None,
        w=None,
        W=None,
    ):
        Tinit = 50
        DTRLS = 2
        tauV = 1e-2 * second  # decay for voltage in spiking net
        taus = 1e-1 * second  # decay for slow synapses
        tauf = 5e-3 * second  # decay for fast synapses
        vth = 1e-16  # spike threshold
        vr = -10  # reset voltage

        np.random.seed(1)
        Jmu = (
            self.muf * (1 / (tauf / second)) * 1 / self.N * np.ones((self.N, self.N))
        )  # mean recurrent matrix in factor approximating network for spiking neurons
        J0f = self.gf * 1 / np.sqrt(self.N) * np.random.randn(self.N, self.N)
        J0s = self.gs * 1 / np.sqrt(self.N) * np.random.randn(self.N, self.N)
        # J0f_mat = scipy.io.loadmat('./J0f_randn_seed_1.mat')
        # J0s_mat = scipy.io.loadmat('./J0s_randn_seed_1.mat')

        # J0f = J0f_mat['J0f']
        # J0s = J0s_mat['J0s']
        #
        # J0f = np.asarray(J0f)
        # J0s = np.asarray(J0s)

        if task in ["reaching"]:
            m = emg.shape[2] + 1
            P = V.shape[1]
        else:
            print("task is not reaching")

        if mode in ["demean"]:
            sum_vJsbar = np.zeros((self.N, 1))
            vJsbar_input = np.zeros((self.N, 1))
            nMSE = np.full((T, 1), np.nan)

        elif mode in ["train", "test", "data"]:
            # Initialize nMSE with NaN values
            nMSE = np.full((T, 1), np.nan)

            if mode in ["train"]:
                W = np.zeros((m, P + 1))  # for computing learned output matrix
                PW = 1 * np.eye(P + 1)  # inverse covariance matrix

                w = np.zeros((P, 2 * self.N))  # P readout weights, half of J
                Pw = 1 * np.eye(2 * self.N)  # inverse covariance matrix

        # Initalize state variables

        # initialize random seed, to set network into same initial state
        # go to different initial state it testing or gathering data
        np.random.seed(3)

        y = np.zeros((P, 1))  # learned input current

        # v = 1e-3 * np.random.randn(self.N, 1)  # spike net state
        v_init = np.asarray(
            1e-3 * np.random.randn(self.N, 1)
        ).flatten()  # spike net state
        # v_mat = scipy.io.loadmat('v_randn_seed_3.mat')
        # v_init = v_mat['v']
        # v_init = np.asarray(v_init).flatten()

        # set various counters
        ttrial = np.inf  # time in current trial
        TTrial = 0  # total time in current trial
        t_trial_num = 0  # trial number
        ttime = 1  # time across all trials

        if task in ["cycling"]:
            fin, TTrial = trial_cycling(self.dt)
            ytilde = factors.T
            f = emg.T

        if mode in ["test"]:
            np.random.seed(4)

        defaultclock.dt = self.dt

        neurons = NeuronGroup(
            self.N,
            model="""
                              vJsbar : 1
                              vinf : 1
                              dv/dt = (v - vinf) * (-1 / tauV) : 1
                              dJ0ss/dt = -J0ss/taus : 1
                              dJ0fs/dt = -J0fs/tauf : 1
                              dJ0fbars/dt = -J0fbars/tauf : 1
                              dss/dt = -ss/taus : 1
                              dsf/dt = -ss/tauf : 1
                              sum_vJsbar : 1
                              nMSE : 1 (shared)
                              """,
            method="euler",
            threshold="v>=vth",
            reset="""
                              v=vr
                              ss += 1
                              sf += 1
                              """,
        )

        neurons.v = v_init
        neurons.J0ss = np.zeros(self.N)  # J0 slow input current
        neurons.J0fs = np.zeros(self.N)  # J0 fast input current
        neurons.J0fbars = np.zeros(self.N)  # J0 mean current
        neurons.sum_vJsbar = np.zeros(self.N)
        neurons.vJsbar = vJsbar_input.flatten()
        neurons.ss = np.zeros(self.N)
        neurons.sf = np.zeros(self.N)
        neurons.nMSE = np.nan

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
        )
        synapses.connect()

        synapses.w1 = J0s.flatten()
        synapses.w2 = J0f.flatten()
        synapses.w3 = Jmu.flatten()

        spikemon = SpikeMonitor(neurons[:20])
        statemon_nMSE = StateMonitor(neurons, "nMSE", record=[0])

        self_u = self.u
        self_u_in = self.u_in

        zs = np.zeros((m, TTrial))  # for collecting z(t), for plotting
        ys = np.zeros((P, TTrial))  # for collecting y(t), for plotting
        if mode in ["demean"]:
            zsT = np.zeros(
                (self.N, TTrial)
            )  # for collecting recurrent inputs for mean computation

        ttime = 1
        _, x, _, _, _ = rate_model(
            emg,
            self.Ntilde,
            m,
            self.dt,
            self.utilde,
            self.utilde_in,
            self.Jtilde,
            x=None,
        )

        @network_operation(dt=defaultclock.dt)
        def trial(t):
            nonlocal \
                ttime, \
                ttrial, \
                t_trial_num, \
                TTrial, \
                zs, \
                ys, \
                zsT, \
                Pw, \
                w, \
                PW, \
                W, \
                task, \
                mode, \
                self_u, \
                y, \
                self_u_in, \
                x, \
                fin, \
                ytilde, \
                f
            # print(f"ttrial = {ttrial} and TTrial = {TTrial}")
            if ttrial >= TTrial:
                # print(f"ttrial = {ttrial} >= TTrial = {TTrial}")
                ttrial = 0

                if t_trial_num == 1:
                    xprimes, x, fin, f, TTrial = rate_model(
                        emg,
                        self.Ntilde,
                        m,
                        self.dt,
                        self.utilde,
                        self.utilde_in,
                        self.Jtilde,
                        x=None,
                    )
                    # print(x.shape)
                else:
                    xprimes, x, fin, f, TTrial = rate_model(
                        emg,
                        Ntilde=self.Ntilde,
                        m=m,
                        dt=self.dt,
                        utilde=self.utilde,
                        utilde_in=self.utilde_in,
                        Jtilde=self.Jtilde,
                        x=x,
                    )

                ytilde = V.T @ xprimes

                zs = np.zeros((m, TTrial))  # for collecting z(t), for plotting
                ys = np.zeros((P, TTrial))  # for collecting y(t), for plotting

                if mode in ["demean"]:
                    zsT = np.zeros(
                        (self.N, TTrial)
                    )  # for collecting recurrent inputs for mean computation

            # def get_fin(t):
            #     event = [0.05, 1.95]
            #     if np.remainder(t / defaultclock.dt, np.sum(event) * second / defaultclock.dt) * defaultclock.dt < \
            #             event[0] * second:
            #         return 2.0
            #     else:
            #         return 0.0

            # print("print shape:")
            # print(neurons.vinf.shape)
            # print(neurons.J0fbars.shape)
            # print(neurons.vJsbar.shape)
            # print(np.dot(self_u, y).flatten().shape)
            # print(neurons.J0ss.shape)
            # print(neurons.J0fs.shape)
            # print(self_u_in.shape)
            # print(fin.shape)
            # print(np.dot(self_u_in, fin).flatten().shape)
            # print(ttrial)
            neurons.vinf = (
                neurons.J0fbars
                - neurons.vJsbar
                + np.dot(self_u, y).flatten()
                + neurons.J0ss
                + neurons.J0fs
                + (np.dot(self_u_in, fin[:, ttrial])).flatten()
            )
            s = np.concatenate((neurons.ss, neurons.sf))
            # ttrial = np.remainder(t/defaultclock.dt, TTrial/defaultclock.dt)
            if t_trial_num > Tinit and mode in ["train", "test", "data"]:
                # print(neurons.ss.shape)
                # print(neurons.sf.shape)
                # print(s.shape)
                y = np.dot(w, s)  # w @ s
                # print(W.shape)
                # print(y.shape)
                # print(np.append(y, 1).shape)
                z = np.dot(W, np.append(y, 1))  # W @ np.vstack((y, np.ones((1, 1))))
            else:
                # print(ttrial)
                # print(ytilde[:, ttrial].shape)
                y = ytilde[:, ttrial]  # use factor targets as feedback
                y = y.reshape(-1, 1)  # Reshape y to (P, 1)
                z = f[:, ttrial]

            # print(zs.shape)
            zs[:, ttrial] = z.flatten()  # output
            # print(ys.shape)
            ys[:, ttrial] = y.flatten()  # factors

            if mode in ["demean"]:
                zsT[:, ttrial] = (
                    np.dot(self.u, ytilde[:, ttrial]) + neurons.J0ss + neurons.J0fs
                )

            if t_trial_num > Tinit:
                if mode in ["train"] and np.random.rand() < 1 / DTRLS:
                    s = np.concatenate((neurons.ss, neurons.sf)).reshape(-1, 1)
                    xP = np.dot(Pw, s).reshape(-1, 1)

                    # s = np.reshape(s, (-1, 1))  # (1600, 1) 形状的二维数组
                    # xP = np.reshape(xP, (-1, 1))  # (1600, 1) 形状的二维数组

                    k = np.dot(
                        np.linalg.inv(1 + np.dot(np.transpose(s), xP)), np.transpose(xP)
                    )
                    Pw = Pw - np.dot(xP, k)
                    w = w - np.dot((y - ytilde[:, ttrial]).reshape(-1, 1), k)

                    y_with_bias = np.append(y, 1).reshape(-1, 1)
                    xP = np.dot(PW, y_with_bias).reshape(-1, 1)
                    k = np.dot(
                        np.linalg.inv(1 + np.dot(np.transpose(y_with_bias), xP)),
                        np.transpose(xP),
                    )
                    PW = PW - np.dot(xP, k)
                    W = W - np.dot((z - f[:, ttrial]).reshape(-1, 1), k)

            ttrial += 1

            if ttrial > TTrial:
                t_trial_num += 1
                if (
                    t_trial_num > Tinit
                ):  # text output, plot things, perform computations
                    if task in ["CI"]:
                        pass
                    else:
                        # nMSE[t_trial_num - Tinit - 1] = np.sum(np.square(ys - ytilde)) / np.sum(np.square(ytilde))
                        # neurons.nMSE = np.sum(np.square(ys - ytilde)) / np.sum(np.square(ytilde))
                        nMSE = np.sum(
                            np.diag((ys - ytilde) @ (ys - ytilde).T)
                        ) / np.sum(np.diag(ytilde @ ytilde.T))

                    if mode == "demean":
                        neurons.sum_vJsbar = (
                            neurons.sum_vJsbar
                            + np.sum(zsT, axis=1, keepdims=True).flatten()
                        )
                        neurons.vJsbar = neurons.sum_vJsbar / (ttime * zsT.shape[1])

                        ttime += 1

                # plot
                if mode in {"demean"}:
                    print(f"{mode}, {t_trial_num - Tinit} trials of {T}")
                else:
                    if task == "CI":
                        print(
                            f"{mode}, {np.nanmean(nMSE):.2g} frac. correct, {t_trial_num - Tinit} trials of {T}"
                        )
                    else:
                        nMSE_filtered = nMSE[~np.isnan(nMSE)]
                        print(
                            f"{mode}, {np.median(nMSE_filtered):.2g} Error, {t_trial_num - Tinit} trials of {T}"
                        )

        # @network_operation(dt=defaultclock.dt * TTrial)
        # def RLS(t):
        #     nonlocal zs, ys, zsT, t_trial_num, ttime, task, mode
        #     t_trial_num += 1
        #     N = 800
        #     # t_trial_num = np.remainder(t / defaultclock.dt, TTrial / (defaultclock.dt * TTrial))
        #     # ttime = t_trial_num - Tinit
        #     if t_trial_num > Tinit:  # text output, plot things, perform computations
        #
        #         if task in ['CI']:
        #             pass
        #         else:
        #             # nMSE[t_trial_num - Tinit - 1] = np.sum(np.square(ys - ytilde)) / np.sum(np.square(ytilde))
        #             # neurons.nMSE = np.sum(np.square(ys - ytilde)) / np.sum(np.square(ytilde))
        #             nMSE = np.sum(np.diag((ys - ytilde) @ (ys - ytilde).T)) / np.sum(
        #                 np.diag(ytilde @ ytilde.T))
        #
        #         if mode == 'demean':
        #             neurons.sum_vJsbar = neurons.sum_vJsbar + np.sum(zsT, axis=1, keepdims=True).flatten()
        #             neurons.vJsbar = neurons.sum_vJsbar / (ttime * zsT.shape[1])
        #
        #             ttime += 1
        #
        #     # plot
        #     if mode in {'demean'}:
        #         print(f'{mode}, {t_trial_num - Tinit} trials of {T}')
        #     else:
        #         if task == 'CI':
        #             print(f'{mode}, {np.nanmean(nMSE):.2g} frac. correct, {t_trial_num - Tinit} trials of {T}')
        #         else:
        #             nMSE_filtered = nMSE[~np.isnan(nMSE)]
        #             print(f'{mode}, {np.median(nMSE_filtered):.2g} Error, {t_trial_num - Tinit} trials of {T}')

        run((T + Tinit) * 2 * second, report="text")

        if mode == "train":
            import matplotlib
            from matplotlib.ticker import MaxNLocator

            matplotlib.use("TkAgg")

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

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
            for output_index in range(zs.shape[0]):
                axes[1].plot(range(zs.shape[1]), zs[output_index, :])
                axes[1].plot(range(f.shape[1]), f[output_index, :], linestyle="--")

            axes[2].set_title("Error")
            axes[2].plot(statemon_nMSE.t / second, statemon_nMSE.nMSE[0])

            axes[3].set_title("Factors")
            for factor_index in range(ys.shape[0]):
                axes[3].plot(range(ys.shape[1]), ys[factor_index, :])
                axes[3].plot(
                    range(ytilde.shape[1]), ytilde[factor_index, :], linestyle="--"
                )

            axes[3].set_xlabel("t [s]")
            fig.tight_layout()
            plt.show()

        if mode == "demean":
            return vJsbar_input
        elif mode == "train":
            return w, W
        elif mode == "test":
            return nMSE
        return -1


def orth(A, tol=None):
    # Perform singular value decomposition
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if tol is None:
        tol = max(A.shape) * np.finfo(s.dtype).eps * np.max(s)

    # Determine the rank based on the given tolerance
    rank = np.sum(s > tol)

    # Return the orthonormal basis
    Q = U[:, :rank]

    return Q


def rate_model(emg, Ntilde, m, dt, utilde, utilde_in, Jtilde, x=None):
    # global Ntilde, m, dt, utilde, Jtilde

    F = np.tanh  # Nonlinearity
    tau = 1e-2 * second
    etau = np.exp(-dt / tau)

    # Initialize state variables
    ttrial = np.inf
    TTrial = 0
    go = True

    if x is None:
        x = 1e0 * np.random.randn(Ntilde, 1).flatten()

    while go:
        if ttrial > TTrial:
            ttrial = 1
            fin, f, TTrial = trial_reaching(dt, emg)
            xprimes = np.zeros((Ntilde, TTrial))

        xprime = x
        # # print(f"x.shape: {x.shape}")
        # # print(f"xprime.shape: {np.asarray(xprime).shape}")
        # print(f"np.dot(Jtilde, x).shape: {(np.dot(Jtilde, x)).shape}")
        # print(f"np.dot(utilde, f[:, ttrial - 1]).shape: {np.dot(utilde, f[:, ttrial - 1]).shape}")
        # print(f"np.dot(utilde_in, fin[:, ttrial - 1]).shape: {np.dot(utilde_in, fin[:, ttrial - 1]).shape}")
        xinf = F(
            (np.dot(Jtilde, x.T)).flatten()
            + np.dot(utilde, f[:, ttrial - 1])
            + np.dot(utilde_in, fin[:, ttrial - 1])
        )
        # print(f"xinf.shape: {xinf.shape}")
        x = xinf + (x - xinf) * etau
        # print(f"x.shape: {x.shape}")
        # print(f"xprime.shape: {np.asarray(xprime).shape}")
        # print(f"xprimes.shape: {xprimes.shape}")

        xprimes[:, ttrial - 1] = xprime

        if ttrial == TTrial:
            go = False

        ttrial += 1

    return xprimes, x, fin, f, TTrial


# Define the trial_reaching function
def trial_reaching(dt, EMG):
    C = EMG.shape[0]
    Nout = EMG.shape[2]

    cond = np.random.choice(C)
    event = np.round(np.array([0.35, 0.15, 0.81 - 0.5]) * (1 / (dt / second))).astype(
        int
    )
    ITIb = int(0.4 * (1 / dt))
    ITI = int(2.0 * (1 / dt))
    ITI = 2 * int(0.5 * (ITIb + np.round(np.random.exponential(ITI))))

    TTrial = sum(event) + ITI
    ttrialb = [ITI // 2 - ITIb // 2 + 1, sum(event) + ITI // 2 + ITIb // 2]

    # Define elements of task input
    fin0 = np.zeros(ITI // 2)
    fin1 = 0.5 * np.cos((cond / C) * 2 * np.pi) * np.ones(sum(event[:3]))
    fin2 = 0.5 * np.sin((cond / C) * 2 * np.pi) * np.ones(sum(event[:3]))
    fin3 = np.ones(sum(event[:1]))

    fout1 = (
        -0.0
        + 2
        * np.reshape(
            np.tile(np.reshape(np.squeeze(EMG[cond, :, :]), -1), (10, 1)).T, (-1, Nout)
        ).T
    )
    fout1[0, :] = -1 * fout1[0, :]
    fout0 = -0.0 * np.ones(ITI // 2)
    fout2 = -0.5 * np.ones(ITI // 2)

    # Concatenate
    fin = np.vstack(
        [
            np.hstack([fin0, fin3, np.zeros(sum(event[1:])), fin0]),
            np.hstack([fin0, fin1, fin0]),
            np.hstack([fin0, fin2, fin0]),
        ]
    )

    fout = np.vstack(
        [
            np.hstack([fout0, fout1[0, :], fout0]),
            np.hstack([fout0, fout1[1, :], fout0]),
            np.hstack([fout2, -0.5 * np.ones(sum(event)), fout2]),
        ]
    )

    return fin, fout, TTrial


def reaching_demo():
    N = 1200  # N = 800  # number of neurons in the spiking network
    Ntilde = 800
    dt = 1e-3 * second  # integration timestep

    sn = SpikingNetwork()
    sn.N = N
    sn.dt = dt
    sn.Ntilde = Ntilde

    data = scipy.io.loadmat("./reaching_data.mat")
    emg = data["emg"]

    m_in = 3  # m_in = 1  # number of inputs
    m = 3  # emg.shape[1]  # number of outputs
    # P = factors.shape[1]  # number of factors

    np.random.seed(1)  # np.random.seed(3)
    # utilde_in = -1 + 2 * np.random.rand(sn.N, m_in)  # input weights for the external input

    utilde_in = (
        -1 + 2 * np.random.rand(m_in, Ntilde).T
    )  # utilde_in = -1 + 2 * np.random.rand(m_in, sn.N).T
    # V = np.linalg.qr(-1 + 2 * np.random.rand(sn.N, P))[
    #     0]  # mapping from factors to N space. In this case, it's just an orthnormal mapping

    gtilde = 1.4
    Jtilde = gtilde * 1 / np.sqrt(Ntilde) * np.random.randn(Ntilde, Ntilde)
    utilde = -1 + 2 * np.random.rand(Ntilde, m)

    sn.utilde = utilde
    sn.utilde_in = utilde_in
    sn.Jtilde = Jtilde

    TPCA = 300
    XX = np.zeros((Ntilde, Ntilde))

    for i in range(TPCA):
        if i == 0:
            xprimes, x, fin, f, TTrial = rate_model(
                emg,
                Ntilde=Ntilde,
                m=m,
                dt=dt,
                utilde=utilde,
                utilde_in=utilde_in,
                Jtilde=Jtilde,
            )
        else:
            xprimes, x, fin, f, TTrial = rate_model(
                emg,
                Ntilde=Ntilde,
                m=m,
                dt=dt,
                utilde=utilde,
                utilde_in=utilde_in,
                Jtilde=Jtilde,
                x=x,
            )
        XX += xprimes @ xprimes.T
        print(f"PCA, {i + 1} trials of {TPCA}")

    # Eigen decomposition and PCA
    d, V = np.linalg.eigh(XX)
    V = np.fliplr(V)
    d = np.flipud(d)

    fracPCA = 0.99
    P = np.argmax(np.cumsum(d) / np.sum(d) > fracPCA) + 1
    V = V[:, :P]

    # A = -1 + 2 * np.random.rand(sn.N, P)
    # A = -1 + 2 * np.random.rand(P, sn.N).T
    # V = orth(A)
    # V[:, [2, 4, 5, 6, 9, 10]] *= -1

    # Parameters for spiking model, and some matrices to the spiking model
    g = 3  # feedback gain in spiking network
    sn.muf = -0.3  # recurrent inhibition
    sn.gs = 0.11  # gain of slow random synapses
    sn.gf = 0.13  # gain of fast random synpases

    U = g * orth((-1 + 2 * np.random.rand(N, Ntilde)))
    sn.u_in = U @ utilde_in  # input weights for the external input to the spiking model
    sn.u = U @ Jtilde @ V  # mapping from P-space to N-space
    print(sn.u.shape)
    # demean spiking inputs
    Tdemean = 300
    vJsbar = sn.spiking_network(task="reaching", mode="demean", T=Tdemean, V=V, emg=emg)
    np.save("vJsbar_reaching_TPCA_300.npy", vJsbar)
    # vJsbar = np.load('vJsbar_reaching.npy')
    # # Train with RLS
    TRLS = 300
    w, W = sn.spiking_network(
        task="reaching", mode="train", T=TRLS, V=V, emg=emg, vJsbar_input=vJsbar
    )
    #
    # # Test
    # Ttest = 100
    # ERR_RLS = sn.spiking_network(task='cycling', mode='test', T=Ttest, factors=factors, emg=emg, vJsbar_input=vJsbar,
    #                              w=w,
    #                              W=W)
    #
    # np.savez('spiking_network_results.npz', vJsbar=vJsbar, w=w, W=W, ERR_RLS=ERR_RLS)


reaching_demo()
