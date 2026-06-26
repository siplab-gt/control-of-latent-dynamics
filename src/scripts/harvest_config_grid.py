"""Aggregate per-cell config_snapshot.yaml files into a flat config grid CSV.

Each model_fit cell writes a ``config_snapshot.yaml`` (the exact merged config it
trained with: swept wildcards + base COLD config + resolved USID config; see
src/scripts/model_fit.py::_dump_config_snapshot). This script collects them across a
results subtree into one row-per-cell table — the realized analog of usid's grid.csv.

Drift-free by construction: it flattens whatever keys each snapshot has and unions
columns across cells, so any new key added to config_params.yaml / cold_usid.yaml /
the sweep .yaml appears as a new column automatically — there is no hardcoded schema
to maintain. The config + sweep YAMLs stay the single source of truth; this is a pure
derived view.

Usage:
    python src/scripts/harvest_config_grid.py <results_subtree> [-o config_grid.csv]
    # e.g. python src/scripts/harvest_config_grid.py results/task_reaching
"""
import argparse
import glob
import os

import pandas as pd
import yaml


def flatten(obj, prefix=""):
    """Flatten a nested dict/list into {dotted.key: scalar}. Lists are repr'd whole
    so a cell stays one row (sweeps vary by scalar wildcards, not list contents)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        out[prefix.rstrip(".")] = repr(list(obj))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def build_grid(root):
    files = sorted(glob.glob(os.path.join(root, "**", "config_snapshot.yaml"), recursive=True))
    rows = []
    for f in files:
        try:
            snap = yaml.safe_load(open(f)) or {}
        except Exception as e:  # noqa: BLE001 - skip a corrupt snapshot, don't abort the harvest
            print(f"  [skip] {f}: {e}")
            continue
        row = flatten(snap)
        row["_cell_dir"] = os.path.dirname(f)
        rows.append(row)
    # pd.DataFrame unions the keys across heterogeneous rows -> drift-free columns.
    return pd.DataFrame(rows), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results subtree to scan for config_snapshot.yaml")
    ap.add_argument("-o", "--out", default=None, help="output CSV (default: <root>/config_grid.csv)")
    args = ap.parse_args()

    df, n = build_grid(args.root)
    if df.empty:
        print(f"No config_snapshot.yaml found under {args.root}")
        return
    out = args.out or os.path.join(args.root, "config_grid.csv")
    df.to_csv(out, index=False)
    print(f"{n} cells -> {out}  ({len(df.columns)} columns)")


if __name__ == "__main__":
    main()
