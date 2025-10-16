import brian2cuda
from brian2 import *

set_device("cuda_standalone", build_on_run=False)
stim = TimedArray(np.zeros(10), dt=10 * ms)
group = NeuronGroup(
    10, "dv/dt = (stim(t) - v)/ (10*ms) : 1", name="bob"
)  # time-dependent stimulus
mon = StateMonitor(group, "v", record=True)
net = Network(group, mon)
print(net["bob"])
run(100 * ms)
device.build(run=True)
results = []
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
