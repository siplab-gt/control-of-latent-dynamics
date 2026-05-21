import logging
import os

import numpy as np
from scipy.stats import bootstrap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from statsmodels.nonparametric.kernel_density import KDEMultivariate

logger = logging.getLogger(__name__)

try:
    import torch
    from tabpfn import TabPFNRegressor
except ImportError:
    pass


class PredictiveCMIRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor_type, eps=1e-6):
        self.regressor_type = regressor_type.lower()
        self.eps = eps
        self.model = None
        self.sigma_ = None
        self.min_sigma_ = None
        self.max_sigma_ = None

    def fit(self, X, y):

        if self.regressor_type == "tabpfn":
            self.model = TabPFNRegressor(device="auto")
            self.model.fit(X, y)

        elif self.regressor_type == "gp":
            # Phase 5: Anisotropy (ARD)
            n_features = X.shape[1]
            kernel = RBF(length_scale=np.ones(n_features)) + WhiteKernel()
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            self.model.fit(X, y)

        elif self.regressor_type == "splines":
            # Phase 1: Splines via LinearRegression + PolynomialFeatures
            # or pygam. We use PolynomialFeatures here.
            base_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
            if len(y) >= 5:
                cv = KFold(n_splits=5, shuffle=True, random_state=0)
                y_pred_oof = cross_val_predict(base_model, X, y, cv=cv)
                self.sigma_ = np.sqrt(mean_squared_error(y, y_pred_oof))
            else:
                base_model.fit(X, y)
                y_pred_train = base_model.predict(X)
                self.sigma_ = np.sqrt(mean_squared_error(y, y_pred_train))
            self.model = base_model
            self.model.fit(X, y)

        elif self.regressor_type == "xgboost":
            from xgboost import XGBRegressor

            base_model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                random_state=0,
                n_jobs=-1,
            )
            if len(y) >= 5:
                cv = KFold(n_splits=5, shuffle=True, random_state=0)
                y_pred_oof = cross_val_predict(base_model, X, y, cv=cv)
                self.sigma_ = np.sqrt(mean_squared_error(y, y_pred_oof))
            else:
                base_model.fit(X, y)
                y_pred_train = base_model.predict(X)
                self.sigma_ = np.sqrt(mean_squared_error(y, y_pred_train))
            self.model = base_model
            self.model.fit(X, y)

        elif self.regressor_type == "xgboost_gauss":
            from xgboost import XGBRegressor

            y = np.asarray(y).ravel()
            y_std = float(np.std(y) + self.eps)
            self.min_sigma_ = max(self.eps, y_std * 0.1)
            self.max_sigma_ = max(self.min_sigma_, y_std * 10.0)
            y_mean = float(np.mean(y))
            y_log_std = float(np.log(y_std))
            base_margin = np.tile([y_mean, y_log_std], (len(y), 1))
            y_targets = np.column_stack([y, np.zeros_like(y)])
            min_log_sigma = float(np.log(self.min_sigma_))
            max_log_sigma = float(np.log(self.max_sigma_))
            log_sigma_prior = float(np.log(y_std))
            sigma_prior_weight = 1.0

            n_estimators = int(os.getenv("XGB_GAUSS_N_ESTIMATORS", 300))
            max_depth = int(os.getenv("XGB_GAUSS_MAX_DEPTH", 4))
            learning_rate = float(os.getenv("XGB_GAUSS_LEARNING_RATE", 0.05))
            subsample = float(os.getenv("XGB_GAUSS_SUBSAMPLE", 0.8))
            reg_lambda = float(os.getenv("XGB_GAUSS_REG_LAMBDA", 10.0))
            reg_alpha = float(os.getenv("XGB_GAUSS_REG_ALPHA", 0.0))
            sigma_prior_weight = float(
                os.getenv("XGB_GAUSS_SIGMA_PRIOR_WEIGHT", sigma_prior_weight)
            )

            def _gauss_obj(labels, preds):
                y_true, preds = _reshape_gauss_labels_preds(labels, preds)
                mu = preds[:, 0]
                log_sigma = np.clip(preds[:, 1], min_log_sigma, max_log_sigma)
                sigma2 = np.exp(2.0 * log_sigma)
                diff = mu - y_true

                grad_mu = diff / sigma2
                hess_mu = np.ones_like(grad_mu) / sigma2

                grad_log_sigma = (
                    1.0
                    - (diff**2) / sigma2
                    + sigma_prior_weight * (log_sigma - log_sigma_prior)
                )
                hess_log_sigma = 2.0 * (diff**2) / sigma2 + sigma_prior_weight

                grad = np.column_stack([grad_mu, grad_log_sigma])
                hess = np.column_stack([hess_mu, hess_log_sigma])
                return grad, hess

            self.model = XGBRegressor(
                objective=_gauss_obj,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                # don't subsample features, since there are only 2 or 3
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                random_state=0,
                n_jobs=-1,
                num_target=2,
                tree_method="hist",
            )
            self.model.fit(
                X,
                y_targets,
                base_margin=base_margin,
                eval_set=[(X, y_targets)],
                verbose=False,
            )

        else:
            raise ValueError(f"Unknown regressor_type: {self.regressor_type}")

        return self

    def predict(self, X):

        if self.regressor_type == "tabpfn":
            tabpfn_out = self.model.predict(X, output_type="full")
            y_mean = tabpfn_out["mean"]

            def log_p_fn(y):
                y_torch = torch.tensor(y, dtype=torch.float32).to(
                    tabpfn_out["logits"].device
                )
                return (
                    np.log(
                        tabpfn_out["criterion"]
                        .pdf(tabpfn_out["logits"], y_torch)
                        .cpu()
                        .numpy()
                    )
                    .squeeze()
                    .astype(float)
                )
        elif self.regressor_type == "gp":
            y_mean, y_std = self.model.predict(X, return_std=True)
            y_std = y_std + self.eps

            def log_p_fn(y):
                return gaussian_logp(y, y_mean, y_std)
        elif self.regressor_type == "splines":
            y_mean = self.model.predict(X)
            y_std = np.full(y_mean.shape, self.sigma_) + self.eps

            def log_p_fn(y):
                return gaussian_logp(y, y_mean, y_std)

        elif self.regressor_type == "xgboost":
            y_mean = self.model.predict(X)
            y_std = np.full(y_mean.shape, self.sigma_) + self.eps

            def log_p_fn(y):
                return gaussian_logp(y, y_mean, y_std)

        elif self.regressor_type == "xgboost_gauss":
            preds = self.model.predict(X)
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 2)
            y_mean = preds[:, 0]
            min_sigma = self.min_sigma_ if self.min_sigma_ is not None else self.eps
            max_sigma = self.max_sigma_ if self.max_sigma_ is not None else np.inf
            log_sigma = np.clip(preds[:, 1], np.log(min_sigma), np.log(max_sigma))
            y_std = np.exp(log_sigma) + self.eps

            def log_p_fn(y):
                return gaussian_logp(y, y_mean, y_std)

        return y_mean, log_p_fn


def gaussian_logp(y, mu, sigma):
    # Evaluate Log-Probability Density using Gaussian likelihood
    return -0.5 * np.log(2 * np.pi * sigma**2) - ((y - mu) ** 2) / (2 * sigma**2)


def _reshape_gauss_labels_preds(labels, preds):
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 2)

    labels = np.asarray(labels)
    if labels.ndim == 1 and labels.shape[0] == preds.size:
        labels = labels.reshape(-1, 2)

    y_true = labels[:, 0] if labels.ndim > 1 else labels
    return y_true, preds


def calculate_total_log_likelihood(X, y, regressor_type, cv=None, cv_groups=None):
    """
    Phase 2: The Estimation Engine (10-fold CV)
    """
    if cv is None:
        if cv_groups is None:
            raise ValueError("cv_groups is required when cv is None")
        unique_groups = np.unique(cv_groups)
        n_groups = len(unique_groups)
        if n_groups < 2:
            raise ValueError("cv_groups must include at least two unique groups")
        cv = GroupKFold(n_splits=n_groups)

    y_flat = y.ravel()

    def fold_log_likelihood(split):
        train_idx, test_idx = split
        X_train = X[train_idx]
        y_train = y_flat[train_idx]

        X_test = X[test_idx]
        y_test = y_flat[test_idx]

        model = PredictiveCMIRegressor(regressor_type=regressor_type)
        model.fit(X_train, y_train)

        _, log_p_fn = model.predict(X_test)
        log_p = log_p_fn(y_test)
        return float(np.sum(log_p))

    fold_ll = np.fromiter(
        map(fold_log_likelihood, cv.split(X, groups=cv_groups)), dtype=float
    )
    return float(np.sum(fold_ll))


def substr(s, ii):
    return "".join([s[j] for j in ii])


def heuristic_bw(data_col, var_type):
    assert len(data_col.shape) == 1
    n = len(data_col)
    if var_type == "c":
        # Silverman's rule of thumb
        sigma = np.std(data_col)
        iqr = np.percentile(data_col, 75) - np.percentile(data_col, 25)
        if iqr == 0:
            a = sigma
            assert sigma > 0
        else:
            a = min(sigma, iqr / 1.34)
        return 0.9 * a * n ** (-1 / 5)
    elif var_type == "o":
        return 1 / (1 + n**0.5)
    elif var_type == "u":
        c = len(np.unique(data_col))
        h = (c - 1) / (c * n**0.5)
        assert 0 <= h <= (c - 1) / c
        return h
    else:
        raise ValueError(f"Unknown var_type: {var_type}")


def kernel_cmi(X, y, Z, var_types, bw=None) -> np.ndarray:
    """Calculate the conditional mutual information of the data.

    Returns
    -------
    local_mi_values : array
        The mutual information between the two datasets.
    """
    # Combine data into a joint dataset
    data = np.hstack([X, y, Z])
    iz = np.arange(2, data.shape[1])
    if bw is None:
        bw = np.array(
            [heuristic_bw(data[:, i], var_types[i]) for i in range(data.shape[1])]
        )
    # logger.debug(f"{bw=}")

    # Compute densities for all points in the joint dataset
    kde_xyz = KDEMultivariate(data, var_types, bw)
    # logger.debug(f"{kde_xyz.bw=}")
    kde_z = KDEMultivariate(data[:, iz], substr(var_types, iz), bw[iz])
    # logger.debug(f"{kde_z.bw=}")
    ixz = [0, *iz]
    kde_xz = KDEMultivariate(data[:, ixz], substr(var_types, ixz), bw[ixz])
    # logger.debug(f"{kde_xz.bw=}")
    iyz = [1, *iz]
    kde_yz = KDEMultivariate(data[:, iyz], substr(var_types, iyz), bw[iyz])
    # logger.debug(f"{kde_yz.bw=}")

    p_xyz = kde_xyz.pdf()
    p_z = kde_z.pdf()
    p_xz = kde_xz.pdf()
    p_yz = kde_yz.pdf()
    for p in [p_xyz, p_z, p_xz, p_yz]:
        assert np.all(p > 0)

    local_mi_values = np.log2(p_xyz) + np.log2(p_z) - np.log2(p_xz) - np.log2(p_yz)
    return np.mean(local_mi_values)


def calculate_cmi(
    X, y, Z, estimator_type=None, cat_XZ=None, cat_Z=None, var_types=None, **kwargs
):
    """
    Phase 3: The CMI Calculation
    Estimates Conditional Mutual Information I(X;Y|Z) in bits.
    """
    if estimator_type in ["tabpfn", "splines", "gp", "xgboost", "xgboost_gauss"]:
        n = len(y)

        cv_groups = kwargs.get("cv_groups", None)
        if cv_groups is None:
            raise ValueError("cv_groups is required for regression CMI estimators")
        unique_groups = np.unique(cv_groups)
        n_groups = len(unique_groups)
        if n_groups < 2:
            raise ValueError("cv_groups must include at least two unique groups")
        cv = GroupKFold(n_splits=n_groups)

        # Full Model
        XZ = np.hstack([X, Z])
        ll_full = calculate_total_log_likelihood(
            XZ, y, estimator_type, cv=cv, cv_groups=cv_groups
        )

        # Reduced Model
        ll_reduced = calculate_total_log_likelihood(
            Z, y, estimator_type, cv=cv, cv_groups=cv_groups
        )

        # Information Gain
        cmi_bits = (ll_full - ll_reduced) / (n * np.log(2))
    elif estimator_type == "kernel":
        cmi_bits = kernel_cmi(X, y, Z, var_types, bw=kwargs.get("bw", None))
    else:
        raise ValueError(f"Unknown CMI estimation method: {estimator_type}")

    assert np.isfinite(cmi_bits)
    return cmi_bits


def _preprocess_features(X, Z, cat_XZ, cat_Z, estimator_type):
    if estimator_type in ["tabpfn"]:
        return X, Z, cat_XZ, cat_Z
    X_proc = X.copy()
    Z_proc = Z.copy()

    cat_X = [c for c in (cat_XZ or []) if c < X.shape[1]]

    # 1. Ordinal Encoding for Continuous features
    for i in range(X.shape[1]):
        if i not in cat_X:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            col = X[:, i : i + 1]
            if not np.issubdtype(col.dtype, np.number):
                pass
            X_proc[:, i : i + 1] = encoder.fit_transform(col)

    for i in range(Z.shape[1]):
        if i not in (cat_Z or []):
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            col = Z[:, i : i + 1]
            if not np.issubdtype(col.dtype, np.number):
                pass
            Z_proc[:, i : i + 1] = encoder.fit_transform(col)

    # 2. OneHotEncoding or OrdnalEncoding for Categorical Features
    if (cat_XZ and len(cat_XZ) > 0) or (cat_Z and len(cat_Z) > 0):
        if len(cat_X) > 0:
            encoder_X = (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                # statsmodels automatically takes care of discrete variables; don't want one-hot
                if estimator_type != "kernel"
                else OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            )
            X_cat = X_proc[:, cat_X]
            X_cat_encoded = encoder_X.fit_transform(X_cat)

            cont_X = [i for i in range(X.shape[1]) if i not in cat_X]
            if len(cont_X) > 0:
                X_cont = X_proc[:, cont_X]
                X_proc = np.hstack([X_cont, X_cat_encoded])
            else:
                X_proc = X_cat_encoded

        cat_Z_list = cat_Z or []
        if len(cat_Z_list) > 0:
            encoder_Z = (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                if estimator_type != "kernel"
                else OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            )
            Z_cat = Z_proc[:, cat_Z_list]
            Z_cat_encoded = encoder_Z.fit_transform(Z_cat)

            cont_Z = [i for i in range(Z.shape[1]) if i not in cat_Z_list]
            if len(cont_Z) > 0:
                Z_cont = Z_proc[:, cont_Z]
                Z_proc = np.hstack([Z_cont, Z_cat_encoded])
            else:
                Z_proc = Z_cat_encoded

    return X_proc, Z_proc, cat_XZ, cat_Z


def statistical_validation_permutation(
    x,
    y,
    z,
    group_ids,
    estimator_type,
    rng,
    cat_XZ=None,
    cat_Z=None,
    var_types=None,
    cv_groups=None,
    n_permutations=1000,
):
    """
    Phase 4: Statistical Validation - Blocked Permutation (p-value).
    """
    kwargs = {
        "estimator_type": estimator_type,
        "random_state": rng,
        "cat_XZ": cat_XZ,
        "cat_Z": cat_Z,
        "var_types": var_types,
    }

    x, z, cat_XZ, cat_Z = _preprocess_features(x, z, cat_XZ, cat_Z, estimator_type)

    if estimator_type == "kernel":
        data = np.hstack([x, y, z])
        bw = np.array(
            [heuristic_bw(data[:, i], var_types[i]) for i in range(data.shape[1])]
        )
        kwargs["bw"] = bw
    kwargs["cv_groups"] = cv_groups
    observed_cmi = calculate_cmi(x, y, z, **kwargs)
    # assert observed_cmi > 0, f"Observed CMI should be non-negative. Got {observed_cmi}"
    logger.info(f"Observed CMI: {observed_cmi}")

    null_cmis = np.zeros(n_permutations)
    unique_groups = np.unique(group_ids)

    logger.info(f"Beginning {n_permutations} permutations...")
    for k in range(n_permutations):
        y_shuffled = np.empty_like(y)
        # Identify groups and shuffle Y inside
        for gid in unique_groups:
            idx = np.where(group_ids == gid)[0]
            shuffled_idx = rng.permutation(idx)
            y_shuffled[idx] = y[shuffled_idx]

        cmi_null = calculate_cmi(x, y_shuffled, z, **kwargs)
        null_cmis[k] = cmi_null

        if k % max(1, n_permutations // 10) == 0:
            logger.info(
                f"Permutations progress: {k}/{n_permutations}. Current CMI: {cmi_null}"
            )

    p_value = np.sum(null_cmis >= observed_cmi) / n_permutations

    return p_value, observed_cmi, null_cmis


def statistical_validation_bootstrap(
    x,
    y,
    z,
    estimator_type,
    rng,
    cat_XZ=None,
    cat_Z=None,
    var_types=None,
    n_boot=1000,
    group_ids=None,
    cluster_ids=None,
    cv_groups=None,
    **kwargs,
):
    """
    Phase 4: Statistical Validation - Bootstrapping (95% CI).
    Because per-group sample sizes are low, we do cluster bootstrapping.
    And since bootstrapping assumes IID samples, we treat each seed as a cluster.
    """
    x, z, cat_XZ, cat_Z = _preprocess_features(x, z, cat_XZ, cat_Z, estimator_type)

    if cluster_ids is None:
        cluster_ids = group_ids

    boot_call_count = [0]

    if estimator_type == "kernel":
        data = np.hstack([x, y, z])
        bw = np.array(
            [heuristic_bw(data[:, i], var_types[i]) for i in range(data.shape[1])]
        )
        kwargs["bw"] = bw

    def _stat(cluster_idx):
        x_boot = []
        y_boot = []
        z_boot = []
        group_ids_boot = []
        cv_groups_boot = []

        for i_cluster in cluster_idx:
            idx = np.where(cluster_ids == i_cluster)[0]
            x_boot.append(x[idx])
            y_boot.append(y[idx])
            z_boot.append(z[idx])
            group_ids_boot.append(group_ids[idx])
            if cv_groups is not None:
                cv_groups_boot.append(cv_groups[idx])

        x_boot = np.concatenate(x_boot, axis=0)
        y_boot = np.concatenate(y_boot, axis=0)
        z_boot = np.concatenate(z_boot, axis=0)
        group_ids_boot = np.concatenate(group_ids_boot, axis=0)
        if cv_groups is not None:
            cv_groups_boot = np.concatenate(cv_groups_boot, axis=0)
        assert len(x_boot) == len(y_boot) == len(z_boot) == len(x)

        cmi = calculate_cmi(
            x_boot,
            y_boot,
            z_boot,
            estimator_type,
            cat_XZ=cat_XZ,
            cat_Z=cat_Z,
            var_types=var_types,
            cv_groups=cv_groups_boot if cv_groups is not None else group_ids_boot,
            **kwargs,
        )
        c = boot_call_count[0]
        if c % max(1, n_boot // 10) == 0:
            logger.info(f"Bootstrap progress: {c}/{n_boot}. Current CMI: {cmi} bits...")
        boot_call_count[0] += 1
        return cmi

    unique_clusters = np.unique(cluster_ids)
    data = (unique_clusters,)
    logger.info(
        f"Beginning {n_boot} bootstrap resamples over {len(unique_clusters)} clusters..."
    )
    res = bootstrap(
        data,
        _stat,
        n_resamples=n_boot,
        method="percentile",  # since BCa supposedly not robust for super small G
        random_state=rng,
    )

    ci_low = np.atleast_1d(res.confidence_interval.low)[0]
    ci_high = np.atleast_1d(res.confidence_interval.high)[0]
    assert np.isfinite(ci_low) and np.isfinite(ci_high)

    boot_cmis = np.asarray(res.bootstrap_distribution)
    return ci_low, ci_high, boot_cmis
