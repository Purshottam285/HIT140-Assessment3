"""
HIT140 Objective 2 - fit and evaluate the two linear regression models.

Run build_wc_data.py FIRST so that matches_104.csv and teams_208.csv exist.

Outputs:
    - full regression summaries printed to screen
    - the numbers you need for Tables 2-5 of the report, printed tidily
    - figure_1.png and figure_2.png (predicted vs actual)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")                      # write files, don't open a window
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# The 8 explanatory variables for each model.
# Exactly 4 appear in both lists - the maximum the brief allows.
# ----------------------------------------------------------------------

X_2_1 = ["rank_diff", "gs_pm_diff", "gc_pm_diff", "pts_pm_diff",   # unique to 2.1
         "stage", "host_adv", "rest_diff", "same_confed"]          # shared

X_2_2 = ["team_rank", "opp_rank", "team_gs_pm_sf", "opp_gc_pm_sf", # unique to 2.2
         "stage", "is_host", "team_rest", "is_uefa"]               # shared

SHARED = {"stage", "host_adv", "rest_diff", "same_confed"} & set(X_2_2) | \
         {"stage", "is_host", "team_rest", "is_uefa"} & set(X_2_1)


def fit(df, x_cols, y_col, title):
    """Fit one OLS model and print everything the report needs."""

    missing = [c for c in x_cols + [y_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"{title}: these columns are missing -> {missing}")

    data = df[x_cols + [y_col]].dropna()
    if len(data) < len(df):
        print(f"!! {title}: dropped {len(df) - len(data)} rows with blanks.")
        print("   Usually means the rank/confed dictionaries are incomplete.\n")

    X = sm.add_constant(data[x_cols])       # the intercept term
    y = data[y_col]

    model = sm.OLS(y, X).fit()

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(model.summary())

    # --- the bits that go straight into your report tables ---
    print(f"\n--- for Table: {title} coefficients ---")
    out = pd.DataFrame({
        "coefficient": model.params.round(4),
        "std error":   model.bse.round(4),
        "t":           model.tvalues.round(3),
        "p-value":     model.pvalues.round(4),
    })
    out["significant"] = np.where(out["p-value"] <= 0.05, "YES", "no")
    print(out.to_string())

    print(f"\n--- for Table: {title} fit statistics ---")
    print(f"R2           : {model.rsquared:.4f}")
    print(f"Adjusted R2  : {model.rsquared_adj:.4f}")
    print(f"F-statistic  : {model.fvalue:.3f}")
    print(f"Prob (F)     : {model.f_pvalue:.6f}")
    print(f"n            : {int(model.nobs)}")
    print()

    return model, data, y


def scatter(model, data, y, x_cols, fname, title, xlabel):
    """Predicted vs actual, with the 'perfect prediction' diagonal."""
    pred = model.predict(sm.add_constant(data[x_cols]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, y, s=28, alpha=0.65, color="#0F6E6A", edgecolor="white", linewidth=0.5)

    lo, hi = min(pred.min(), y.min()), max(pred.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#C97A1F", linewidth=1.4,
            label="perfect prediction")

    ax.set_xlabel(f"Predicted {xlabel}")
    ax.set_ylabel(f"Actual {xlabel}")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"saved {fname}")


if __name__ == "__main__":
    m104 = pd.read_csv("matches_104.csv")
    m208 = pd.read_csv("teams_208.csv")

    print(f"loaded {len(m104)} matches and {len(m208)} team-matches\n")

    mod1, d1, y1 = fit(m104, X_2_1, "goal_diff",
                       "MODEL 2.1 - goal difference (104 matches)")
    mod2, d2, y2 = fit(m208, X_2_2, "goals_for",
                       "MODEL 2.2 - goals scored (208 team-matches)")

    scatter(mod1, d1, y1, X_2_1, "figure_1.png",
            "Model 2.1: predicted vs actual goal difference", "goal difference")
    scatter(mod2, d2, y2, X_2_2, "figure_2.png",
            "Model 2.2: predicted vs actual goals scored", "goals scored")

    print("\nShared variables between the two models:",
          sorted(set(X_2_1) & set(X_2_2)) or "(none by name - see report note)")
