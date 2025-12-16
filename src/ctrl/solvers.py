from dataclasses import dataclass
from functools import cache


@dataclass
class Solver:
    name: str
    solve_kwargs: dict
    ctrl_type_specific_solve_kwargs: dict

    @cache
    def get_solve_kwargs(self, ctrl_type: str) -> dict:
        return {
            "solver": self.name,
            **self.solve_kwargs,
            **self.ctrl_type_specific_solve_kwargs.get(ctrl_type, {}),
        }


OL_MAX_ITERS = int(1e6)
OL_TIME_LIMIT = 2  # representative of inter-trial interval for reaching
EPS_INFEAS = 1e-10

SHARED_OL = dict(
    verbose=True,
    ignore_dpp=True,
)
SHARED_LQMPC = dict(
    warm_start=True,
    enforce_dpp=True,
)

SOLVERS = [
    Solver(
        "osqp",
        dict(
            eps_prim_inf=EPS_INFEAS,
            eps_dual_inf=EPS_INFEAS,
        ),
        dict(
            OL=SHARED_OL
            | dict(
                max_iter=OL_MAX_ITERS,
                time_limit=OL_TIME_LIMIT,
            ),
            LQMPC=SHARED_LQMPC | dict(),
        ),
    ),
    Solver(
        "clarabel",
        dict(
            tol_infeas_abs=EPS_INFEAS,
            tol_infeas_rel=EPS_INFEAS,
        ),
        dict(
            OL=SHARED_OL
            | dict(
                max_iter=OL_MAX_ITERS,
                time_limit=OL_TIME_LIMIT,
            ),
            LQMPC=SHARED_LQMPC | dict(),
        ),
    ),
    Solver(
        "scs",
        dict(
            eps_infeas=EPS_INFEAS,
        ),
        dict(
            OL=SHARED_OL
            | dict(
                max_iters=OL_MAX_ITERS,  # note: max_iters (with 's')
                time_limit_secs=OL_TIME_LIMIT,
            ),
            LQMPC=SHARED_LQMPC | dict(),
        ),
    ),
]
