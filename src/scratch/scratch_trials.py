import matplotlib.pyplot as plt
import numpy as np
from src.scripts.ctrl_expts import CtrlData
from src.scripts.example_figs import _trial_slice

# Load data
paths = [
    "results/ctrl_data_q1_k5.pkl.gz",
    "results/ctrl_data_q1_k100.pkl.gz",
    "results/ctrl_data_q1_k1024.pkl.gz"
]

cds = [CtrlData.load(p) for p in paths]
ref_arr = cds[0].Xref_fsnn
conds = np.array(cds[0].conditions)
bounds = cds[0].i_trial_boundaries
n_dims = min(8, ref_arr.shape[1])

fig, axs = plt.subplots(4, 8, figsize=(24, 12), sharex=True, sharey=True)

arrays = [ref_arr] + [cd.X for cd in cds]
labels = ["Reference", "k=5", "k=100", "k=1024"]

for row, (arr, label) in enumerate(zip(arrays, labels)):
    for cond in range(8):
        ax = axs[row, cond]
        idx = np.where(conds == cond)[0]
        if len(idx) == 0:
            ax.axis("off")
            continue
            
        for i in idx:
            trial_data = _trial_slice(arr, bounds, int(i))
            for dim in range(n_dims):
                # Each trial as a translucent line
                ax.plot(trial_data[:, dim], color=f"C{dim}", alpha=0.1, lw=0.5)
                
        if row == 0:
            ax.set_title(f"Cond {cond + 1}")
        if cond == 0:
            ax.set_ylabel(label)

plt.tight_layout()
plt.savefig("results/figs/scratch_trials.webp")
print("Saved to results/figs/scratch_trials.webp")
