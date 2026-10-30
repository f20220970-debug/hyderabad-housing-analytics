"""Price prediction.

1. Multivariate regression (OLS on log price): interpretable drivers. log(price) on log(area), bedrooms,
   resale, amenity score and a locality price level (target-encoded inside each CV fold to avoid leakage).
   Coefficients read as % effects. Variance inflation factors check for overlapping predictors.
2. Gradient boosting (HistGradientBoostingRegressor) on the same features plus individual amenities:
   usually more accurate, less explainable.
Both are compared with 5-fold cross-validation: MAE in INR lakh, median absolute % error, and R^2 on log price.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 7


def locality_level(train: pd.DataFrame, test: pd.DataFrame, k: float = 10.0) -> tuple:
    """Smoothed mean log price-per-sqft by locality, fitted on train only (shrinks small localities to the city mean)."""
    lp = np.log(train["price_per_sqft"])
    g = lp.groupby(train["location"]).agg(["mean", "count"])
    city = lp.mean()
    enc = (g["mean"] * g["count"] + city * k) / (g["count"] + k)
    return train["location"].map(enc).fillna(city).to_numpy(), test["location"].map(enc).fillna(city).to_numpy()


def features(df, loc_level, amen=None):
    X = pd.DataFrame({"log_area": np.log(df["area"]).to_numpy(), "bedrooms": df["bedrooms"].to_numpy(),
                      "resale": df["resale"].fillna(0).to_numpy(), "amenity_score": df["amenity_score"].to_numpy(),
                      "locality_level": loc_level})
    if amen:
        X = pd.concat([X, df[amen].fillna(-1).reset_index(drop=True)], axis=1)
    return X


def cross_validate(df, amen, folds=5):
    y = np.log(df["price"].to_numpy())
    res = {"ols": [], "gbm": []}
    preds = np.zeros((len(df), 2))
    for tr, te in KFold(folds, shuffle=True, random_state=SEED).split(df):
        a, b = df.iloc[tr], df.iloc[te]
        la, lb = locality_level(a, b)
        Xa, Xb = features(a, la), features(b, lb)
        ols = sm.OLS(y[tr], sm.add_constant(Xa)).fit()
        preds[te, 0] = ols.predict(sm.add_constant(Xb, has_constant="add"))
        gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED)
        gbm.fit(features(a, la, amen), y[tr])
        preds[te, 1] = gbm.predict(features(b, lb, amen))
    out = []
    for i, name in enumerate(["Multivariate regression (OLS)", "Gradient boosting"]):
        p = np.exp(preds[:, i]); a = df["price"].to_numpy()
        out.append({"model": name, "mae_lakh": np.mean(np.abs(p - a)) / 1e5,
                    "median_ape": np.median(np.abs(p - a) / a), "r2_log": 1 - np.sum((y - preds[:, i]) ** 2) / np.sum((y - y.mean()) ** 2)})
    return pd.DataFrame(out)


def drivers(df):
    """Full-sample OLS for interpretation, with % effects and VIFs."""
    lvl, _ = locality_level(df, df)
    X = features(df, lvl)
    fit = sm.OLS(np.log(df["price"]), sm.add_constant(X)).fit()
    coef = fit.params.drop("const")
    tab = pd.DataFrame({"coef": coef, "p_value": fit.pvalues.drop("const")})
    tab["effect"] = [f"{100 * (np.exp(c) - 1):+.1f}% per unit" if k != "log_area" else f"{c:.2f}% per 1% more area"
                     for k, c in coef.items()]
    vif_X = sm.add_constant(X)
    tab["vif"] = [variance_inflation_factor(vif_X.values, i) for i in range(1, vif_X.shape[1])]
    return tab, fit.rsquared
