from dataclasses import dataclass
from functools import cache

argname_map = {
    "osqp": {
        "eps_infeas": ["eps_prim_inf", "eps_dual_inf"],
    },
    "clarabel": {
        "eps_infeas": ["tol_infeas_abs", "tol_infeas_rel"],
    },
    "scs": {
        "max_iter": "max_iters",
        "time_limit": "time_limit_secs",
    },
}
"""Maps generic solver argument names to solver-specific names when necessary"""


@cache
def _generic_to_specific_args(solver_name: str, generic_args: frozenset) -> dict:
    specific_args = {"solver": solver_name}
    for k, v in generic_args:
        k_mapped = argname_map.get(solver_name, {}).get(k, k)
        if isinstance(k_mapped, list):
            for kk in k_mapped:
                specific_args[kk] = v
        else:
            specific_args[k_mapped] = v
    return specific_args


@dataclass
class Solver:
    name: str
    generic_args: dict

    @property
    def specific_args(self) -> dict:
        """kwargs with solver-specific argument names"""
        return _generic_to_specific_args(
            self.name, frozenset(self.generic_args.items())
        )


def get_solvers(solvers: list[str], cfg: dict) -> list[Solver]:
    if "max_iter" in cfg:
        cfg["max_iter"] = int(cfg["max_iter"])
    return [Solver(name=name, generic_args=cfg) for name in solvers]
