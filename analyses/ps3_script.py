# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
# %%
# %%
from ps3.data import load_transform

print("Loading OpenML data...", flush=True)
df = load_transform()
print(df.shape)
print(df[["IDpol", "Exposure", "ClaimAmountCut"]].head())



# %%
from ps3.data import create_sample_split

df = create_sample_split(df, id_column="IDpol", training_frac=0.9)

df_train = df.loc[df["sample"] == "train"].copy()
df_test = df.loc[df["sample"] == "test"].copy()

print(df["sample"].value_counts())



# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def make_glm(numeric_features):
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_features),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", TweedieRegressor(
            power=POWER, link="log", alpha=0.0, max_iter=1000
        )),
    ])


def deviance(model, X, y, exposure):
    return mean_tweedie_deviance(
        y, model.predict(X), sample_weight=exposure, power=POWER
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


def neg_weighted_deviance(model, X_val, y_val):
    weights = w_train.loc[X_val.index]
    return -deviance(model, X_val, y_val, weights)


def cv_deviance(model, X):
    scores = cross_validate(
        model,
        X,
        y_train,
        cv=cv,
        scoring=neg_weighted_deviance,
        params={"model__sample_weight": w_train},
        error_score="raise",
    )
    return -scores["test_score"]


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

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

lgbm_preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
])

lgbm_pipeline = Pipeline([
    ("preprocess", lgbm_preprocess),
    ("model", LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=POWER,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )),
])

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

results = pd.DataFrame({
    "Model": ["Linear Tweedie", "Quadratic Tweedie", "Tuned LightGBM"],
    "Train deviance": [linear_train, quadratic_train, lgbm_train],
    "CV deviance": [linear_cv.mean(), quadratic_cv.mean(), -search.best_score_],
    "Test deviance": [linear_test, quadratic_test, lgbm_test],
})

print("Best LightGBM parameters:", search.best_params_)
print(results.round(4).to_string(index=False))

# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

df_test["PurePremium"] = y_test
df_test["pp_t_glm1"] = linear_glm.predict(X_test)
df_test["pp_t_glm2"] = quadratic_glm.predict(X_test_quad)
df_test["pp_t_lgbm"] = best_lgbm.predict(X_test)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.arange(1, len(cumulated_claim_amount) + 1)
    cumulated_samples = cumulated_samples / len(cumulated_claim_amount)
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