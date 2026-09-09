# COLD configuration

Every run takes a **pair** of configfiles, merged by Snakemake — a params file and a sweeps file. The
Snakefile validates both at parse time and refuses to run without them:

```bash
sbatch smk-submit-slurm.sbatch ../config/<name>_params.yaml ../config/<name>_sweeps.yaml
```

- **`<name>_params.yaml`** — one default for every condition parameter, plus the per-rule blocks. The
  Snakefile asserts that each cond-param registered in `steps_params` has a default here, so adding a
  wildcard means adding a default to all three params files.
- **`<name>_sweeps.yaml`** — the `sweeps:` block listing the axes to fan out, as a full Cartesian
  product. Every key must already be a recognized cond-param.

Three pairs are committed: `config` (production), `testconfig` (full-length quality, seed roster only)
and `ultra_light_config` (a wiring smoke test whose fits are intentionally bad). Alongside them,
`cold_usid.yaml` holds the USID training hyperparameters COLD's `model_fit` rule uses, and
`minimal_cache.yaml` and `brian_preferences/` are support files.

**One-off sweeps do not get committed here.** Copy a sweeps file into `tmp/`, add your axes, and delete
it when the experiment concludes. Keep the `_sweeps` token in the filename — the rule-input blocklist
filters on it, and a name without it is tracked as a per-cell input, so touching the file re-runs every
cell.
