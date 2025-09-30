import types

import numpy as np

from src.helpers import seed_everything
from src.model_fit import LDS


def fit_lds_to_data(
    YN, ZN, UN, X_fsnn, n_x, config, cond_params, base_params, outfiles
) -> tuple[LDS, np.ndarray, np.ndarray, np.ndarray]:
    pass
    seed_everything(int(cond_params["seed"]), "model_fit")

    X_minus = X_fsnn[:-1]
    X_plus = X_fsnn[1:]
    # estimate matrices via least square
    A, _, _, _ = np.linalg.lstsq(X_minus, X_plus)
    A = A.T

    B, _, _, _ = np.linalg.lstsq(UN[:-1], X_plus - X_minus @ A.T)
    B = B.T

    C, _, _, _ = np.linalg.lstsq(X_fsnn, YN)
    C = C.T

    Cz, _, _, _ = np.linalg.lstsq(X_fsnn, ZN)
    Cz = Cz.T

    # estimate Q, R
    Q = np.cov((X_plus - X_minus @ A.T - UN[:-1] @ B.T), rowvar=False)
    R = np.cov((YN - X_fsnn @ C.T), rowvar=False)

    k_opto = int(cond_params["k"])
    lds = LDS(A=A, B=B, C=C, Q=Q, R=R, k_opto=k_opto)

    Zhat = X_fsnn @ Cz.T
    Yhat = X_fsnn @ C.T
    Xhat = np.concatenate([X_fsnn[:1], X_fsnn[:-1] @ A.T + UN[:-1] @ B.T], axis=0)

    # just use linear Cz for now. May want to do MLP or something later
    lds.Cz = Cz

    def x2z(self, X):
        return X @ self.Cz.T

    lds.x2z = types.MethodType(x2z, lds)

    return lds, Zhat, Yhat, Xhat
