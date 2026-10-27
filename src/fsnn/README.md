# fSNN data attribution

These data come from [briandepasquale/factor-based-spiking-nets at commit
fac6eae9c5bb2ffdb883c4ef45a6f282dcc9aa8a](https://github.com/briandepasquale/factor-based-spiking-nets/tree/fac6eae9c5bb2ffdb883c4ef45a6f282dcc9aa8a):

- [cycling_data.npz](cycling_data.npz) was converted from
  [cycling/cycling_data.mat](https://github.com/briandepasquale/factor-based-spiking-nets/blob/fac6eae9c5bb2ffdb883c4ef45a6f282dcc9aa8a/cycling/cycling_data.mat).
- [reaching_data.npz](reaching_data.npz) was converted from
  [reaching/reaching_data.mat](https://github.com/briandepasquale/factor-based-spiking-nets/blob/fac6eae9c5bb2ffdb883c4ef45a6f282dcc9aa8a/reaching/reaching_data.mat).

The only change was MATLAB `.mat` to NumPy `.npz` format conversion; array
values are unchanged.

Reference: DePasquale et al. (2023), “The centrality of population-level factors
to network computation is demonstrated by a versatile approach for training
spiking networks”, *Neuron*. https://doi.org/10.1016/j.neuron.2022.12.007

These data are provided under the upstream Apache-2.0 license; see
[LICENSE-data.txt](LICENSE-data.txt). This data notice is distinct from the
project's root [MIT license](../../LICENSE).
