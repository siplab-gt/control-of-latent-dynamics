# %%
import brian2.only as b2
import brian2cuda
import matplotlib.pyplot as plt
from brian2 import np

b2.set_device("cuda_standalone", build_on_run=False)
stim = b2.TimedArray(np.arange(10) + np.arange(10)[:, None], dt=10 * b2.ms)
stim.values /= 10
group = b2.NeuronGroup(
    10,
    """dv/dt = (I + stim(t, i) - v)/ (10*ms) : 1
        I : 1
        w_neuron : 1
        w_tot : 1 (linked)
        w_tot_copy = w_tot : 1 (shared)
    """,
    threshold="v>1",
    reset="v=0",
)
ng2 = b2.NeuronGroup(1, "w_tot : 1")
group.w_tot = b2.linked_var(ng2, "w_tot")
syn = b2.Synapses(
    group,
    group,
    "w : 1 \n w_neuron_post = w : 1 (summed) \n x = w_tot_copy_post * 2 : 1",
    on_pre={"pre_step1": "v += w"},
)
syn.connect("i!=j")
syn.w = 0.1


syn2 = b2.Synapses(
    group,
    ng2,
    "w_tot_post = w_neuron_pre : 1 (summed)",
)
syn2.connect("j==0")
mon = b2.StateMonitor(group, ["v", "w_neuron", 'w_tot_copy'], record=True)

net = b2.Network(group, ng2, syn, syn2, mon)
net.run(100 * b2.ms, profile=True)
b2.device.build(run=True, clean=True)

plt.plot(mon.t / b2.ms, mon.w_tot_copy.T)
plt.gcf()
# Do 10 runs with a 10ms at a random time
# for idx in range(10):
#     values = np.zeros(10)
#     values[np.random.randint(0, 10)] = 1
#     run_args = {stim: values}
#     if idx > 0:
#         print(f"before trial {idx}: {group.v=}")
#         run_args[group.v] = group.v
#     device.run(run_args=run_args)
#     print(f"after trial {idx}: {group.v=}")
#     results.append(mon.v[0])

# %%
# print(b2.profiling_summary(net))

# %%
# b2.scheduling_summary()

# %%
mon.w_tot_copy.shape