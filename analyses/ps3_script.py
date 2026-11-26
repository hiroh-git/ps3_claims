# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
# %%
# %%

print("Loading OpenML data...", flush=True)
df = load_transform()
print(df.shape)
print(df[["IDpol", "Exposure", "ClaimAmountCut"]].head())


# %%

df = create_sample_split(df, id_column="IDpol", training_frac=0.9)

df_train = df.loc[df["sample"] == "train"].copy()
df_test = df.loc[df["sample"] == "test"].copy()

print(df["sample"].value_counts())


# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance

df_train = df_train.loc[df_train["Exposure"] > 0].copy()
df_test = df_test.loc[df_test["Exposure"] > 0].copy()

w_train = df_train["Exposure"]
w_test = df_test["Exposure"]

# Divide by exposure to express claims per insured year.
y_train = df_train["ClaimAmountCut"] / w_train
y_test = df_test["ClaimAmountCut"] / w_test

categorical_cols = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge"]
numeric_cols = ["BonusMalus", "VehPower", "Density"]
feature_cols = categorical_cols + numeric_cols

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

POWER = 1.5


def make_glm(numeric_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                TweedieRegressor(power=POWER, link="log", alpha=0.0, max_iter=1000),
            ),
        ]
    )


def deviance(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, exposure: pd.Series
) -> float:
    return float(
        mean_tweedie_deviance(y, model.predict(X), sample_weight=exposure, power=POWER)
    )


linear_glm = make_glm(numeric_cols)
linear_glm.fit(X_train, y_train, model__sample_weight=w_train)

linear_train = deviance(linear_glm, X_train, y_train, w_train)
linear_test = deviance(linear_glm, X_test, y_test, w_test)

print(f"Linear GLM train deviance: {linear_train:.4f}")
print(f"Linear GLM test deviance:  {linear_test:.4f}")


# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

X_train_quad = X_train.assign(
    BonusMalus_sq=X_train["BonusMalus"] ** 2,
    VehPower_sq=X_train["VehPower"] ** 2,
)
X_test_quad = X_test.assign(
    BonusMalus_sq=X_test["BonusMalus"] ** 2,
    VehPower_sq=X_test["VehPower"] ** 2,
)

quadratic_glm = make_glm(numeric_cols + ["BonusMalus_sq", "VehPower_sq"])
quadratic_glm.fit(X_train_quad, y_train, model__sample_weight=w_train)

quadratic_train = deviance(quadratic_glm, X_train_quad, y_train, w_train)
quadratic_test = deviance(quadratic_glm, X_test_quad, y_test, w_test)

print(f"Quadratic GLM train deviance: {quadratic_train:.4f}")
print(f"Quadratic GLM test deviance:  {quadratic_test:.4f}")
print(f"Change in train deviance: {quadratic_train - linear_train:+.4f}")
print(f"Change in test deviance:  {quadratic_test - linear_test:+.4f}")


# %%
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=3, shuffle=True, random_state=42)


def neg_weighted_deviance(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    weights = w_train.loc[X_val.index]
    return -deviance(model, X_val, y_val, weights)


def cv_deviance(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    scores = cross_validate(
        model,
        X,
        y_train,
        cv=cv,
        scoring=neg_weighted_deviance,
        params={"model__sample_weight": w_train},
        error_score="raise",
    )
    return -np.asarray(scores["test_score"])


linear_cv = cv_deviance(linear_glm, X_train)
quadratic_cv = cv_deviance(quadratic_glm, X_train_quad)
fold_changes = quadratic_cv - linear_cv

print("CV deviance change by fold:", fold_changes.round(4))
print(f"Mean CV deviance change: {fold_changes.mean():+.4f}")


# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.


lgbm_preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

lgbm_pipeline = Pipeline(
    [
        ("preprocess", lgbm_preprocess),
        (
            "model",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=POWER,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            ),
        ),
    ]
)

search = GridSearchCV(
    lgbm_pipeline,
    param_grid={
        "model__learning_rate": [0.03, 0.08],
        "model__n_estimators": [100, 250],
        "model__num_leaves": [8, 16],
    },
    cv=cv,
    scoring=neg_weighted_deviance,
    refit=True,
    verbose=1,
    error_score="raise",
)
search.fit(X_train, y_train, model__sample_weight=w_train)

best_lgbm = search.best_estimator_
lgbm_train = deviance(best_lgbm, X_train, y_train, w_train)
lgbm_test = deviance(best_lgbm, X_test, y_test, w_test)

results = pd.DataFrame(
    {
        "Model": ["Linear Tweedie", "Quadratic Tweedie", "Tuned LightGBM"],
        "Train deviance": [linear_train, quadratic_train, lgbm_train],
        "CV deviance": [linear_cv.mean(), quadratic_cv.mean(), -search.best_score_],
        "Test deviance": [linear_test, quadratic_test, lgbm_test],
    }
)

print("Best LightGBM parameters:", search.best_params_)
print(results.round(4).to_string(index=False))

# %%


df_test["PurePremium"] = y_test
df_test["pp_t_glm1"] = linear_glm.predict(X_test)
df_test["pp_t_glm2"] = quadratic_glm.predict(X_test_quad)
df_test["pp_t_lgbm"] = best_lgbm.predict(X_test)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(
    y_true: ArrayLike, y_pred: ArrayLike, exposure: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    n_samples = len(cumulated_claim_amount)
    cumulated_samples = np.arange(1, n_samples + 1) / n_samples
    return np.r_[0.0, cumulated_samples], np.r_[0.0, cumulated_claim_amount]


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("Tuned LightGBM", df_test["pp_t_lgbm"]),
    ("Linear Tweedie", df_test["pp_t_glm1"]),
    ("Quadratic Tweedie", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.show()

# %%
# Adding quadratic terms reduced train and test deviance by 0.0652 and 0.0358.
# Mean CV deviance fell by 0.0222, with improvement in two of the three folds.
# The tuned LightGBM had the lowest test deviance (72.8520) and the highest
# Gini among the fitted models (0.251).

# %%
# =============================================================================
# PS4: Model evaluation & interpretability
# =============================================================================

# %%
# PS4 Ex 1, Step 1: exposure-weighted average claims per BonusMalus value.
# Training data only, because this plot informs a modelling decision
# (whether to add a monotonicity constraint) and must not see the test set

bm_summary = (
    df_train.groupby("BonusMalus")
    .agg(claim_amount=("ClaimAmountCut", "sum"), exposure=("Exposure", "sum"))
    .assign(avg_claims=lambda d: d["claim_amount"] / d["exposure"])
)

fig, (ax_claims, ax_exposure) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax_claims.plot(
    bm_summary.index, bm_summary["avg_claims"], marker="o", markersize=3, linewidth=1
)
ax_claims.set(
    title="Exposure-weighted average claims per BonusMalus value (training data)",
    ylabel="Claim amount per year of exposure",
)
ax_exposure.bar(bm_summary.index, bm_summary["exposure"], width=1)
ax_exposure.set(xlabel="BonusMalus", ylabel="Exposure (years)", yscale="log")
for ax in (ax_claims, ax_exposure):
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# PS4 Ex 1, Step 2: LGBM with an increasing monotonicity constraint on BonusMalus.
from sklearn.base import clone

# LightGBM sees the columns AFTER preprocessing (one-hot columns + numeric
# passthrough), so the constraint list must follow that order and length.
lgbm_feature_names = clone(lgbm_preprocess).fit(X_train).get_feature_names_out()
monotone_constraints = [
    1 if name == "num__BonusMalus" else 0 for name in lgbm_feature_names
]

constrained_lgbm = clone(lgbm_pipeline).set_params(
    model__monotone_constraints=monotone_constraints
)

print(f"LightGBM sees {len(lgbm_feature_names)} columns")
print(
    "Constrained column(s):",
    [n for n, c in zip(lgbm_feature_names, monotone_constraints) if c == 1],
)

# %%
# PS4 Ex 1, Step 3: cross-validate the constrained LGBM with the SAME grid,
# folds and scoring as the unconstrained one, so only the constraint differs.
constrained_search = GridSearchCV(
    constrained_lgbm,
    param_grid=search.param_grid,
    cv=cv,
    scoring=neg_weighted_deviance,
    refit=True,
    verbose=1,
    error_score="raise",
)
constrained_search.fit(X_train, y_train, model__sample_weight=w_train)

best_constrained_lgbm = constrained_search.best_estimator_
df_test["pp_t_lgbm_constrained"] = best_constrained_lgbm.predict(X_test)

comparison = pd.DataFrame(
    {
        "Model": ["Unconstrained LightGBM", "Constrained LightGBM"],
        "Train deviance": [
            lgbm_train,
            deviance(best_constrained_lgbm, X_train, y_train, w_train),
        ],
        "CV deviance": [-search.best_score_, -constrained_search.best_score_],
        "Test deviance": [
            lgbm_test,
            deviance(best_constrained_lgbm, X_test, y_test, w_test),
        ],
    }
)
print("Best constrained parameters:", constrained_search.best_params_)
print(comparison.round(4).to_string(index=False))

# %%
# Sanity check: does the prediction ever DECREASE when only BonusMalus increases?
bm_grid = np.arange(X_train["BonusMalus"].min(), X_train["BonusMalus"].max() + 1)
check_rows = X_test.sample(200, random_state=42)


def n_decreasing_profiles(model: Pipeline) -> int:
    """Count rows whose prediction falls somewhere along the BonusMalus grid."""
    n_decreasing = 0
    for _, row in check_rows.iterrows():
        profile = pd.DataFrame([row] * len(bm_grid)).assign(BonusMalus=bm_grid)
        n_decreasing += int(np.any(np.diff(model.predict(profile)) < 0))
    return n_decreasing


print("Rows (of 200) with a decreasing BonusMalus profile:")
print("  unconstrained:", n_decreasing_profiles(best_lgbm))
print("  constrained:  ", n_decreasing_profiles(best_constrained_lgbm))

#  small loss in deviance (about 0.2, small compared with the fold-to-fold variation of about 0.8)
#  buys predictions that are monotone in BonusMalus, plus less overfitting

# %%
# PS4 Ex 2: learning curve of the best constrained LGBM.
import lightgbm as lgb

# eval_set data is passed straight to LightGBM, bypassing the pipeline, so we
# preprocess it with the fitted preprocessing step (LightGBM sees 54 columns).
fitted_preprocess = best_constrained_lgbm.named_steps["preprocess"]
X_train_prep = fitted_preprocess.transform(X_train)
X_test_prep = fitted_preprocess.transform(X_test)

# Best constrained settings, but 1000 trees to see where the test curve stops
# improving. Diagnostic only: picking n_estimators from the test curve would
# be tuning on the test set (use CV or early stopping on train data instead).
curve_lgbm = clone(best_constrained_lgbm.named_steps["model"]).set_params(
    n_estimators=1000
)
curve_lgbm.fit(
    X_train_prep,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_train_prep, y_train), (X_test_prep, y_test)],
    eval_names=["train", "test"],
    eval_sample_weight=[w_train, w_test],
)

best_n = constrained_search.best_params_["model__n_estimators"]
ax = lgb.plot_metric(curve_lgbm, figsize=(8, 5))
ax.axvline(best_n, color="grey", linestyle="--", label=f"CV choice ({best_n} trees)")
ax.legend()
plt.show()

test_curve = curve_lgbm.evals_result_["test"]["tweedie"]
print(f"Lowest test metric after {np.argmin(test_curve) + 1} trees")

# The test metric is still falling at the CV choice of 250 trees (the largest
# value in the grid) and only flattens around 600 trees, so the model is not
# tuned optimally: the grid should include more trees. Train keeps falling
# while test flattens, i.e. extra trees mostly fit the training data.
