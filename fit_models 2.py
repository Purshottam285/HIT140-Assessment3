"""
HIT140 Objective 2 - fit and evaluate the two linear regression models.

S371375 Naresh Kumar Chaudhary
S408952 Pawan Koirala
S395858 Mehedi Hasan Limon
S401527 Rohan Rai 
S246353 Shahab Uddin 


Run build_wc_data.py FIRST so matches_104.csv and teams_208.csv exist.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_SIZE = 0.20          # Week 8: 80% training, 20% test
SEED      = 42            # fixed so the split is reproducible

# ----------------------------------------------------------------------
# The 8 explanatory variables for each model (the brief allows max 4 shared)
# ----------------------------------------------------------------------

X_2_1 = ["rank_diff", "gs_pm_diff", "gc_pm_diff", "pts_pm_diff",
         "stage", "host_adv", "rest_diff", "same_confed"]

X_2_2 = ["team_rank", "opp_rank", "team_gs_pm_sf", "opp_gc_pm_sf",
         "stage", "is_host", "team_rest", "is_uefa"]


# ----------------------------------------------------------------------
# WEEK 9 - collinearity assumption
# "calculate pairwise correlations between the explanatory variables and
#  remove one from the most correlated pair"
# ----------------------------------------------------------------------

def check_collinearity(X, title, threshold=0.8):
    corr = X.corr()
    print(f"\n--- {title}: pairwise correlations between explanatory variables ---")
    print(corr.round(2).to_string())

    pairs = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(r, 3)))

    if pairs:
        print(f"\n!! Pairs with |r| >= {threshold} - consider removing one of each:")
        for a, b, r in pairs:
            print(f"   {a} & {b}: r = {r}")
    else:
        print(f"\nNo pair reaches |r| = {threshold}. "
              "The collinearity assumption is satisfied.")
    return corr


# ----------------------------------------------------------------------
# WEEK 6 - outlier detection on the response variable, IQR method
#   outlier  <  Q1 - 1.5*IQR   or   >  Q3 + 1.5*IQR
# Week 9 recommends removing outliers from y, but the brief fixes the
# dataset at exactly 104 and 208 rows, so we DETECT and REPORT instead
# of removing. That choice is stated in the report.
# ----------------------------------------------------------------------

def check_outliers(y, title):
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = y[(y < lo) | (y > hi)]

    print(f"\n--- {title}: outlier check on the response variable (IQR method) ---")
    print(f"Q1 = {q1:.2f}   Q3 = {q3:.2f}   IQR = {iqr:.2f}")
    print(f"Outlier limits: below {lo:.2f} or above {hi:.2f}")
    print(f"Outliers found: {len(out)} of {len(y)} rows "
          f"({100*len(out)/len(y):.1f}%)")
    if len(out):
        print(f"Values: {sorted(out.unique())}")
    print("Retained - the brief requires exactly this many rows.")
    return out


# ----------------------------------------------------------------------
# WEEK 6 - exploratory scatterplot of the strongest predictor vs response
# ----------------------------------------------------------------------

def explore_scatter(X, y, xcol, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[xcol], y, s=30, alpha=0.6, color="#0F6E6A",
               edgecolor="white", linewidth=0.5)
    b, a = np.polyfit(X[xcol], y, 1)          # Week 1 best-fit line
    xs = np.linspace(X[xcol].min(), X[xcol].max(), 50)
    ax.plot(xs, b * xs + a, "--", color="#C97A1F", linewidth=1.5,
            label=f"best fit (r = {X[xcol].corr(y):.2f})")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout(); fig.savefig(fname, dpi=200)
    print(f"saved {fname}")


# ----------------------------------------------------------------------
# WEEK 8 - performance metrics
# ----------------------------------------------------------------------

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # NRMSE normalised by the range of the response variable (Week 8)
    nrmse = rmse / (y_true.max() - y_true.min())

    print(f"\n--- {label} ---")
    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"NRMSE : {nrmse:.4f}   (0 = perfect, 1 = poor)")
    return dict(MAE=mae, MSE=mse, RMSE=rmse, NRMSE=nrmse)


# ----------------------------------------------------------------------
# Fit one model
# ----------------------------------------------------------------------

def fit(df, x_cols, y_col, title):
    missing = [c for c in x_cols + [y_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"{title}: missing columns -> {missing}")

    data = df[x_cols + [y_col]].dropna()
    X, y = data[x_cols], data[y_col]

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    # ---- Week 9 assumption check, on the full dataset ----
    check_collinearity(X, title)

    # ---- Week 6 outlier check on the response variable ----
    check_outliers(y, title)

    # ---- Week 8: split into training and test sets ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"\nTraining set: {len(X_tr)} rows   Test set: {len(X_te)} rows "
          f"({int(TEST_SIZE*100)}% held out)")

    # ---- Week 9: multiple linear regression via scikit-learn ----
    skl = LinearRegression().fit(X_tr, y_tr)

    # ---- statsmodels on the SAME training set, for p-values ----
    sm_model = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    print("\n" + sm_model.summary().as_text())

    # ---- coefficient table for the report ----
    print(f"\n--- Table: {title} coefficients (training set) ---")
    ci = sm_model.conf_int()
    out = pd.DataFrame({
        "coefficient": sm_model.params.round(4),
        "std error":   sm_model.bse.round(4),
        "t":           sm_model.tvalues.round(3),
        "p-value":     sm_model.pvalues.round(4),
        "CI lower":    ci[0].round(4),
        "CI upper":    ci[1].round(4),
    })
    out["significant"] = np.where(out["p-value"] <= 0.05, "YES", "no")
    print(out.to_string())

    # ---- Week 9: R2 and adjusted R2 ----
    r2_tr = skl.score(X_tr, y_tr)
    r2_te = skl.score(X_te, y_te)
    n, p = len(X_te), len(x_cols)
    adj_te = 1 - (1 - r2_te) * (n - 1) / (n - p - 1)

    print(f"\n--- Table: {title} fit statistics ---")
    print(f"R2 (training): {r2_tr:.4f}")
    print(f"R2 (test)    : {r2_te:.4f}")
    print(f"Adjusted R2 (test): {adj_te:.4f}")
    print(f"R2 (statsmodels, training): {sm_model.rsquared:.4f}")
    print(f"Adjusted R2 (statsmodels, training): {sm_model.rsquared_adj:.4f}")
    print(f"F-statistic  : {sm_model.fvalue:.3f}")
    print(f"Prob (F)     : {sm_model.f_pvalue:.6f}")
    print(f"n (total)    : {len(data)}")

    # ---- Week 8: error metrics on the TEST set ----
    metrics(y_te, skl.predict(X_te), f"{title}: TEST set performance")
    metrics(y_tr, skl.predict(X_tr), f"{title}: training set (for comparison)")

    return skl, X_te, y_te


def scatter(model, X_te, y_te, fname, title, label):
    pred = model.predict(X_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, y_te, s=34, alpha=0.7, color="#0F6E6A",
               edgecolor="white", linewidth=0.5)
    lo, hi = min(pred.min(), y_te.min()), max(pred.max(), y_te.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#C97A1F", linewidth=1.4,
            label="perfect prediction")
    ax.set_xlabel(f"Predicted {label}")
    ax.set_ylabel(f"Actual {label}")
    ax.set_title(f"{title}\n(test set, {len(y_te)} rows)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"saved {fname}")


if __name__ == "__main__":
    m104 = pd.read_csv("matches_104.csv")
    m208 = pd.read_csv("teams_208.csv")
    print(f"loaded {len(m104)} matches and {len(m208)} team-matches")

    mod1, Xte1, yte1 = fit(m104, X_2_1, "goal_diff",
                           "MODEL 2.1 - goal difference")
    mod2, Xte2, yte2 = fit(m208, X_2_2, "goals_for",
                           "MODEL 2.2 - goals scored")

    # Week 6: exploratory scatterplots of the strongest predictor
    explore_scatter(m104[X_2_1], m104["goal_diff"], "rank_diff",
                    "figure_3.png", "FIFA ranking difference", "goal difference")
    explore_scatter(m208[X_2_2], m208["goals_for"], "opp_rank",
                    "figure_4.png", "opponent FIFA ranking", "goals scored")

    scatter(mod1, Xte1, yte1, "figure_1.png",
            "Model 2.1: predicted vs actual goal difference", "goal difference")
    scatter(mod2, Xte2, yte2, "figure_2.png",
            "Model 2.2: predicted vs actual goals scored", "goals scored")

    print("\nShared variables by name:", sorted(set(X_2_1) & set(X_2_2)))
