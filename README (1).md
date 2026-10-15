# HIT140 Foundations of Data Science — Assessment 3

**Objective 2: Linear Regression Models, FIFA World Cup 2026**

Charles Darwin University · Semester 2, 2026

### Group members

| Student ID | Name |
|---|---|
| S371375 | Naresh Kumar Chaudhary |
| S408952 | Pawan Koirala |
| S395858 | Mehedi Hasan Limon |
| S401527 | Rohan Rai |
| S246353 | Shahab Uddin |

---

## Overview

Two multiple linear regression models built on FIFA World Cup 2026 data.

| Model | Predicts | Rows | Explanatory variables |
|---|---|---|---|
| **2.1** | Goal difference in a match | 104 (one per match) | 8 |
| **2.2** | Goals scored by a team in a match | 208 (one per team per match) | 8 |

**Key constraint:** every explanatory variable uses information available **before kick-off**. No in-match statistics are used.

---

## Data sources

| Source | Provides |
|---|---|
| [FBref — 2026 World Cup Scores & Fixtures](https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures) | Date, round, home team, away team, score for all 104 matches |
| FIFA World Rankings (published 11 June 2026) | Pre-tournament ranking for all 48 teams |
| Confederation membership | UEFA, CAF, AFC, CONMEBOL, CONCACAF, OFC |

The ranking date matters: 11 June 2026 is the opening day of the tournament, so it is the last ranking published before any match was played.

---

## Repository contents

```
build_wc_data.py      Builds both datasets from the raw fixture list
fit_models.py         Fits and evaluates both regression models
wc_fixtures_raw.txt   Raw FBref fixtures page (source data)
matches_104.csv       Generated — Model 2.1 dataset
teams_208.csv         Generated — Model 2.2 dataset
figure_1.png          Model 2.1 predicted vs actual (test set)
figure_2.png          Model 2.2 predicted vs actual (test set)
figure_3.png          Ranking difference vs goal difference
figure_4.png          Opponent ranking vs goals scored
```

---

## How to run

**Requirements:** Python 3.11+, with:

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
```

**Run in order:**

```bash
python build_wc_data.py wc_fixtures_raw.txt
python fit_models.py
```

Expected output from the first script:

```
matches : 104 rows   (need 104)
teams   : 208 rows   (need 208)
teams seen: 48 (need 48)
```

---

## Explanatory variables

### Model 2.1 — goal difference

Expressed as differences between the two teams.

| Variable | Measures |
|---|---|
| `rank_diff` | Gap in FIFA world ranking |
| `gs_pm_diff` | Gap in goals scored per match so far |
| `gc_pm_diff` | Gap in goals conceded per match so far |
| `pts_pm_diff` | Gap in points per match so far (3 win / 1 draw) |
| `stage` | Tournament round (1 = group … 6 = final) |
| `host_adv` | +1 home team is host, −1 away team is host |
| `rest_diff` | Gap in days since previous match |
| `same_confed` | 1 if both teams share a confederation |

### Model 2.2 — goals scored

Expressed as levels for one team and its opponent.

| Variable | Measures |
|---|---|
| `team_rank` | The team's own FIFA ranking |
| `opp_rank` | The opponent's FIFA ranking |
| `team_gs_pm_sf` | Team's goals per match so far |
| `opp_gc_pm_sf` | Opponent's goals conceded per match so far |
| `stage` | Tournament round |
| `is_host` | 1 if the team is a host nation |
| `team_rest` | Days since the team's previous match |
| `is_uefa` | 1 if the team is a UEFA member |

By exact column name, `stage` is the only variable shared between the models.

---

## Data preparation notes

Four issues in the raw FBref data required handling:

1. **En-dash score separator.** Scores are written `2–0` using U+2013, not an ASCII hyphen. Splitting on `-` fails silently on every row.
2. **Penalty shoot-outs.** Four matches appear as `(3) 1–1 (4)`. The bracketed numbers are the shoot-out, not goals, so goal difference for those matches is 0.
3. **Country codes in team names.** The home column reads `Mexico mx` (code after), the away column `za South Africa` (code before). FBref also repeats its header row mid-table.
4. **Name differences between sources.** FIFA and FBref disagree on six teams — Bosnia–Herz, IR Iran, Korea Republic, Côte d'Ivoire, Congo DR, Cabo Verde.

**A constant variable was found and removed.** The difference in matches played before a fixture is zero in all 104 matches, because both teams in any World Cup fixture have always played the same number of games. `statsmodels` reported a singular design matrix and `Df Model: 7` instead of 8. It was replaced with points-per-match form, which does vary.

**First-match handling.** Before a team's opening match no prior tournament data exists, affecting 48 team-rows. Since the brief requires exactly 104 and 208 rows, these cannot be dropped; all "so far" variables are set to 0 and this is noted as a limitation.

---

## Method

- Multiple linear regression, ordinary least squares
- **80 / 20 train-test split**, `random_state=42` for reproducibility
- `scikit-learn` `LinearRegression` for fitting; `statsmodels` OLS on the same training set for p-values and confidence intervals
- Evaluated on R², adjusted R², MAE, MSE, RMSE and NRMSE
- Assumption checks: pairwise collinearity between explanatory variables, and IQR outlier detection on the response variable

---

## Results summary

| | Model 2.1 | Model 2.2 |
|---|---|---|
| Training / test rows | 83 / 21 | 166 / 42 |
| R² (training) | 0.4786 | 0.3184 |
| Adjusted R² (training) | 0.4223 | 0.2837 |
| **R² (test)** | **0.1276** | **−0.1203** |
| MAE (test) | 1.3592 | 0.9603 |
| RMSE (test) | 1.6630 | 1.2338 |
| NRMSE (test) | 0.2772 | 0.2468 |

**Significant predictors (p ≤ 0.05):**

- Model 2.1 — `rank_diff` only (p < 0.001). `stage` is marginally non-significant at p = 0.0503.
- Model 2.2 — `team_rank`, `opp_rank`, `opp_gc_pm_sf`, `is_uefa`.

**Assumption checks:** no variable pair reaches |r| = 0.8 in either model. Model 2.1 has zero IQR outliers in the response; Model 2.2 has three (1.4%), retained because the brief fixes the row count.

**Main finding:** both models overfit. Test R² falls well below training R², and Model 2.2's is negative — predicting worse than the mean on unseen data. With 83 and 166 training rows against the eight variables the brief mandates, this is structurally likely. FIFA ranking, built over years of results, outperformed in-tournament form as a predictor.

---

## Acknowledgement of AI use

Generative AI (Claude, Anthropic) was used to assist with writing and debugging the Python code in this repository, and to explain statistical concepts used in the analysis. All code has been reviewed, executed and verified by the group. Data selection, variable design and the written report are the group's own work.

This is declared in the AI Usage Declaration Form submitted with the assessment.
