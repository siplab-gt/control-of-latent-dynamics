import logging

import numpy as np
from scipy.stats import bootstrap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

logger = logging.getLogger(__name__)

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    pass


class PredictiveCMIRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor_type="tabpfn", categorical_features=None, eps=1e-6):
        self.regressor_type = regressor_type.lower()
        self.categorical_features = (
            categorical_features if categorical_features is not None else []
        )
        self.eps = eps
        self.model = None
        self.encoder = None
        self.sigma_ = None

    def fit(self, X, y):
        # Determine continuous and categorical parts for non-TabPFN
        if self.regressor_type != "tabpfn" and len(self.categorical_features) > 0:
            self.encoder = OneHotEncoder(sparse_output=False)
            cat_cols = [X[:, i] for i in self.categorical_features]
            X_cat = np.column_stack(cat_cols)
            X_cat_encoded = self.encoder.fit_transform(X_cat)

            cont_indices = [
                i for i in range(X.shape[1]) if i not in self.categorical_features
            ]
            if len(cont_indices) > 0:
                cont_cols = [X[:, i] for i in cont_indices]
                X_cont = np.column_stack(cont_cols)
                X_encoded = np.hstack([X_cont, X_cat_encoded])
            else:
                X_encoded = X_cat_encoded
        else:
            X_encoded = X

        if self.regressor_type == "tabpfn":
            # TabPFN accepts categorical_features directly
            self.model = TabPFNRegressor(categorical_features=self.categorical_features)
            self.model.fit(X, y)

        elif self.regressor_type == "gp":
            # Phase 5: Anisotropy (ARD)
            n_features = X_encoded.shape[1]
            kernel = RBF(length_scale=np.ones(n_features)) + WhiteKernel()
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            self.model.fit(X_encoded, y)

        elif self.regressor_type == "splines":
            # Phase 1: Splines via LinearRegression + PolynomialFeatures
            # or pygam. We use PolynomialFeatures here.
            self.model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
            self.model.fit(X_encoded, y)

            # Phase 1: estimate \sigma using RMSE of the training residuals
            y_pred_train = self.model.predict(X_encoded)
            self.sigma_ = np.sqrt(mean_squared_error(y, y_pred_train))

        else:
            raise ValueError(f"Unknown regressor_type: {self.regressor_type}")

        return self

    def predict(self, X, return_std=True):
        if self.regressor_type != "tabpfn" and self.encoder is not None:
            cat_cols = [X[:, i] for i in self.categorical_features]
            X_cat = np.column_stack(cat_cols)
            X_cat_encoded = self.encoder.transform(X_cat)

            cont_indices = [
                i for i in range(X.shape[1]) if i not in self.categorical_features
            ]
            if len(cont_indices) > 0:
                cont_cols = [X[:, i] for i in cont_indices]
                X_cont = np.column_stack(cont_cols)
                X_encoded = np.hstack([X_cont, X_cat_encoded])
            else:
                X_encoded = X_cat_encoded
        else:
            X_encoded = X

        if self.regressor_type == "tabpfn":
            y_mean, y_std = self.model.predict(X, return_std=True)
            y_std = y_std + self.eps
        elif self.regressor_type == "gp":
            y_mean, y_std = self.model.predict(X_encoded, return_std=True)
            y_std = y_std + self.eps
        elif self.regressor_type == "splines":
            y_mean = self.model.predict(X_encoded)
            y_std = np.full(y_mean.shape, self.sigma_) + self.eps

        if return_std:
            return y_mean, y_std
        return y_mean


def calculate_total_log_likelihood(X, y, regressor_type, categorical_features=None):
    """
    Phase 2: The Estimation Engine (LOO)
    """
    n = len(y)
    total_log_likelihood = 0.0

    for i in range(n):
        # Leave-One-Out Loop
        idx_train = np.arange(n) != i
        X_train = X[idx_train]
        y_train = y[idx_train]

        X_test = X[i : i + 1]
        y_test = y[i : i + 1]

        model = PredictiveCMIRegressor(
            regressor_type=regressor_type, categorical_features=categorical_features
        )
        model.fit(X_train, y_train.ravel())

        mu_i, sigma_i = model.predict(X_test, return_std=True)

        mu = mu_i[0]
        sigma = sigma_i[0]
        y_i = y_test.ravel()[0]

        # Evaluate Log-Probability Density using Gaussian likelihood
        log_prob = -0.5 * np.log(2 * np.pi * sigma**2) - ((y_i - mu) ** 2) / (
            2 * sigma**2
        )
        total_log_likelihood += log_prob

    return total_log_likelihood


def calculate_cmi(X, y, Z, regressor_type, cat_XZ=None, cat_Z=None):
    """
    Phase 3: The CMI Calculation
    Estimates Conditional Mutual Information I(X;Y|Z) in bits.
    """
    n = len(y)

    # Full Model
    XZ = np.hstack([X, Z])
    ll_full = calculate_total_log_likelihood(
        XZ, y, regressor_type, categorical_features=cat_XZ
    )

    # Reduced Model
    ll_reduced = calculate_total_log_likelihood(
        Z, y, regressor_type, categorical_features=cat_Z
    )

    # Information Gain
    cmi_bits = (ll_full - ll_reduced) / (n * np.log(2))

    return cmi_bits, ll_full, ll_reduced


def statistical_validation_permutation(
    X,
    y,
    Z,
    group_ids,
    regressor_type,
    cat_XZ=None,
    cat_Z=None,
    n_permutations=1000,
    random_state=None,
):
    """
    Phase 4: Statistical Validation - Blocked Permutation (p-value).
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    observed_cmi, _, _ = calculate_cmi(X, y, Z, regressor_type, cat_XZ, cat_Z)

    null_cmis = np.zeros(n_permutations)
    unique_groups = np.unique(group_ids)

    for k in range(n_permutations):
        if k % max(1, n_permutations // 10) == 0:
            logger.info(f"Permutations progress: {k}/{n_permutations}")

        y_shuffled = np.empty_like(y)
        # Identify groups and shuffle Y inside
        for gid in unique_groups:
            idx = np.where(group_ids == gid)[0]
            shuffled_idx = rng.permutation(idx)
            y_shuffled[idx] = y[shuffled_idx]

        cmi_null, _, _ = calculate_cmi(X, y_shuffled, Z, regressor_type, cat_XZ, cat_Z)
        null_cmis[k] = cmi_null

    p_value = np.sum(null_cmis >= observed_cmi) / n_permutations

    return p_value, observed_cmi, null_cmis


def statistical_validation_bootstrap(
    X, y, Z, regressor_type, cat_XZ=None, cat_Z=None, n_boot=1000, random_state=None
):
    """
    Phase 4: Statistical Validation - Bootstrapping (95% CI).
    """

    boot_call_count = [0]

    def _stat(idx):
        c = boot_call_count[0]
        if c % max(1, n_boot // 10) == 0:
            logger.info(f"Bootstrap progress: {c}/{n_boot} resamples evaluating...")
        boot_call_count[0] += 1
        return calculate_cmi(X[idx], y[idx], Z[idx], regressor_type, cat_XZ, cat_Z)[0]

    data = (np.arange(len(y)),)
    res = bootstrap(
        data,
        _stat,
        n_resamples=n_boot,
        method="BCa",
        random_state=random_state,
    )

    ci_low = np.atleast_1d(res.confidence_interval.low)[0]
    ci_high = np.atleast_1d(res.confidence_interval.high)[0]

    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        res = bootstrap(
            data,
            _stat,
            n_resamples=n_boot,
            method="percentile",
            random_state=random_state,
        )
        ci_low = np.atleast_1d(res.confidence_interval.low)[0]
        ci_high = np.atleast_1d(res.confidence_interval.high)[0]

    boot_cmis = np.asarray(res.bootstrap_distribution)
    return ci_low, ci_high, boot_cmis
