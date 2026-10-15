"""
HIT140 Objective 2 - dataset builder.

Reads the FBref "2026 World Cup Scores & Fixtures" table (saved as a .txt or .csv)
and produces the two datasets the brief requires:

    matches_104.csv  -> 104 rows, one per match      (Linear Regression 2.1)
    teams_208.csv    -> 208 rows, one per team-match (Linear Regression 2.2)

Every explanatory variable is computed from information available BEFORE kick-off.
"""

import re
import pandas as pd

# ----------------------------------------------------------------------
# 1. READ THE FIXTURE LIST
# ----------------------------------------------------------------------
# FBref rows are tab separated. We keep only lines that start with a round
# name AND contain "Match Report" - that second test is what removes the
# header row FBref repeats in the middle of long tables.

ROUNDS = ("Group stage", "Round of 32", "Round of 16",
          "Quarter-finals", "Semi-finals", "Third-place match", "Final")

COLUMNS = ["Round", "Wk", "Day", "Date", "Time", "Home", "Score", "Away",
           "Attendance", "Venue", "Referee", "MatchReport", "Notes"]

def read_fixtures(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith(ROUNDS) and "Match Report" in line:
            parts = line.split("\t")
            parts += [""] * (len(COLUMNS) - len(parts))      # pad short rows
            rows.append(parts[:len(COLUMNS)])
    return pd.DataFrame(rows, columns=COLUMNS)


# ----------------------------------------------------------------------
# 2. CLEAN THE THREE FBref QUIRKS
# ----------------------------------------------------------------------

def clean_home(name):
    """'Mexico mx' -> 'Mexico'   (country code is AFTER the home team)"""
    return re.sub(r"\s+[a-z]{2,3}$", "", name.strip())

def clean_away(name):
    """'za South Africa' -> 'South Africa'  (code is BEFORE the away team)"""
    return re.sub(r"^[a-z]{2,3}\s+", "", name.strip())

def parse_score(score):
    """
    Returns (home_goals, away_goals) for the match itself.

    Two formats appear:
        '2-0'            normal result
        '(3) 1-1 (4)'    draw that went to a penalty shoot-out

    The bracketed numbers are the SHOOT-OUT score, not goals. A shoot-out
    does not change the goal difference of the match, so we strip the
    brackets and keep 1-1. Those four matches therefore have goal_diff = 0,
    which is correct.

    NOTE: FBref separates the goals with an EN DASH (U+2013), not a hyphen.
    score.split('-') silently fails on every row. This is the single most
    common bug when parsing this table.
    """
    score = re.sub(r"\(\d+\)", "", score).strip()      # drop shoot-out numbers
    home, away = re.split(r"[-–—]", score)             # hyphen OR en/em dash
    return int(home.strip()), int(away.strip())


# ----------------------------------------------------------------------
# 3. BUILD THE BASE TABLE
# ----------------------------------------------------------------------

STAGE_ORDER = {"Group stage": 1, "Round of 32": 2, "Round of 16": 3,
               "Quarter-finals": 4, "Semi-finals": 5,
               "Third-place match": 6, "Final": 6}

HOSTS = {"USA", "Canada", "Mexico"}

def build_base(path):
    df = read_fixtures(path)

    df["home_team"] = df["Home"].map(clean_home)
    df["away_team"] = df["Away"].map(clean_away)
    df[["home_goals", "away_goals"]] = df["Score"].apply(
        lambda s: pd.Series(parse_score(s)))

    df["date"]  = pd.to_datetime(df["Date"])
    df["stage"] = df["Round"].map(STAGE_ORDER)

    # OUTCOME for model 2.1
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index + 1
    return df[["match_id", "date", "Round", "stage",
               "home_team", "away_team", "home_goals", "away_goals", "goal_diff"]]


# ----------------------------------------------------------------------
# 4. PRE-MATCH FORM VARIABLES
# ----------------------------------------------------------------------
# For every match we look ONLY at matches each team already played before
# that date. This is what keeps the variables legal under the brief.

def add_form(df):
    played, scored, conceded, points, last_date = {}, {}, {}, {}, {}
    out = []

    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]

        rec = {}
        for side, team in (("h", h), ("a", a)):
            n = played.get(team, 0)
            rec[f"{side}_played_sf"] = n
            # per-match averages; 0 before a team's first game (see note below)
            rec[f"{side}_gs_pm_sf"]  = scored.get(team, 0) / n if n else 0.0
            rec[f"{side}_gc_pm_sf"]  = conceded.get(team, 0) / n if n else 0.0
            rec[f"{side}_pts_pm_sf"] = points.get(team, 0) / n if n else 0.0
            rec[f"{side}_rest"] = ((m["date"] - last_date[team]).days
                                   if team in last_date else 0)
        out.append(rec)

        # now update the running state with THIS match
        gh, ga_ = m["home_goals"], m["away_goals"]
        # 3 points for a win, 1 for a draw, 0 for a loss
        pts_h = 3 if gh > ga_ else (1 if gh == ga_ else 0)
        pts_a = 3 if ga_ > gh else (1 if gh == ga_ else 0)

        for team, gf, gc, pt in ((h, gh, ga_, pts_h), (a, ga_, gh, pts_a)):
            played[team]   = played.get(team, 0) + 1
            scored[team]   = scored.get(team, 0) + gf
            conceded[team] = conceded.get(team, 0) + gc
            points[team]   = points.get(team, 0) + pt
            last_date[team] = m["date"]

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ----------------------------------------------------------------------
# 5. MODEL 2.1 TABLE - 104 rows, 8 explanatory variables
# ----------------------------------------------------------------------

def build_104(df, rank, confed):
    d = df.copy()
    d["rank_diff"]   = d["away_team"].map(rank) - d["home_team"].map(rank)
    d["gs_pm_diff"]  = d["h_gs_pm_sf"] - d["a_gs_pm_sf"]
    d["gc_pm_diff"]  = d["h_gc_pm_sf"] - d["a_gc_pm_sf"]
    # NOTE: h_played_sf - a_played_sf is ALWAYS 0 in a World Cup - both teams
    # in any fixture have played the same number of matches. It was a constant
    # column, so it is replaced by points-per-match form, which does vary.
    d["pts_pm_diff"] = d["h_pts_pm_sf"] - d["a_pts_pm_sf"]
    d["rest_diff"]   = d["h_rest"] - d["a_rest"]
    d["host_adv"]    = (d["home_team"].isin(HOSTS).astype(int)
                        - d["away_team"].isin(HOSTS).astype(int))
    d["same_confed"] = (d["home_team"].map(confed)
                        == d["away_team"].map(confed)).astype(int)
    # stage already computed
    return d[["match_id", "date", "home_team", "away_team", "goal_diff",
              "rank_diff", "gs_pm_diff", "gc_pm_diff", "pts_pm_diff",
              "rest_diff", "host_adv", "same_confed", "stage"]]


# ----------------------------------------------------------------------
# 6. MODEL 2.2 TABLE - 208 rows, 8 explanatory variables
# ----------------------------------------------------------------------
# Each match becomes TWO rows: once from the home team's point of view,
# once from the away team's.

def build_208(df, rank, confed):
    frames = []
    for side, opp in (("h", "a"), ("a", "h")):
        d = pd.DataFrame({
            "match_id":  df["match_id"],
            "date":      df["date"],
            "team":      df[f"{'home' if side=='h' else 'away'}_team"],
            "opponent":  df[f"{'home' if opp =='h' else 'away'}_team"],
            "goals_for": df["home_goals"] if side == "h" else df["away_goals"],
            "team_gs_pm_sf":  df[f"{side}_gs_pm_sf"],
            "team_pts_pm_sf": df[f"{side}_pts_pm_sf"],
            "opp_gc_pm_sf":   df[f"{opp}_gc_pm_sf"],
            "team_rest":      df[f"{side}_rest"],
            "stage":          df["stage"],
        })
        d["team_rank"] = d["team"].map(rank)
        d["opp_rank"]  = d["opponent"].map(rank)
        d["is_host"]   = d["team"].isin(HOSTS).astype(int)
        d["confed"]    = d["team"].map(confed)
        # Binary flag rather than 6-way dummy coding, so this counts
        # unambiguously as ONE explanatory variable under the brief's
        # "exactly 8" rule. It also reuses the UEFA / non-UEFA split
        # from Assessment 2 Task B.
        d["is_uefa"]   = (d["confed"] == "UEFA").astype(int)
        frames.append(d)

    return (pd.concat(frames)
              .sort_values(["match_id", "team"])
              .reset_index(drop=True))


# ----------------------------------------------------------------------
# 7. RUN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "wc_fixtures_raw.txt"

    # YOUR GROUP SUPPLIES THESE TWO LOOKUPS (48 teams each)
    #   rank   : team name -> FIFA world ranking before the tournament
    #   confed : team name -> 'UEFA' / 'CAF' / 'AFC' / 'CONMEBOL' / 'CONCACAF' / 'OFC'
    rank   = {"Argentina":1,  "Spain":2,   "France":3,   "England":4,  "Portugal":5,
    "Brazil":6,     "Morocco":7, "Netherlands":8, "Belgium":9, "Germany":10,
    "Croatia":11,   "Colombia":13, "Mexico":14, "Senegal":15, "Uruguay":16,
    "USA":17,       "Japan":18,  "Switzerland":19, "IR Iran":20, "Türkiye":22,
    "Ecuador":23,   "Austria":24, "Korea Republic":25, "Australia":27,
    "Algeria":28,   "Egypt":29,  "Canada":30,  "Norway":31,
    "Côte d'Ivoire":33, "Panama":34, "Sweden":38, "Czechia":40,
    "Paraguay":41,  "Scotland":42, "Tunisia":45, "Congo DR":46,
    "Uzbekistan":50, "Qatar":56, "Iraq":57,    "South Africa":60,
    "Saudi Arabia":61, "Jordan":63, "Bosnia–Herz":64, "Cabo Verde":67,
    "Ghana":73,     "Curaçao":82, "Haiti":83,  "New Zealand":85,}          # <- fill from the FIFA rankings page
    confed = {# UEFA (16)
    "England":"UEFA", "France":"UEFA", "Spain":"UEFA", "Belgium":"UEFA",
    "Norway":"UEFA", "Germany":"UEFA", "Netherlands":"UEFA", "Switzerland":"UEFA",
    "Portugal":"UEFA", "Sweden":"UEFA", "Austria":"UEFA", "Croatia":"UEFA",
    "Bosnia–Herz":"UEFA", "Türkiye":"UEFA", "Czechia":"UEFA", "Scotland":"UEFA",
    # CONMEBOL (6)
    "Argentina":"CONMEBOL", "Brazil":"CONMEBOL", "Colombia":"CONMEBOL",
    "Paraguay":"CONMEBOL", "Uruguay":"CONMEBOL", "Ecuador":"CONMEBOL",
    # CONCACAF (6)
    "USA":"CONCACAF", "Mexico":"CONCACAF", "Canada":"CONCACAF",
    "Curaçao":"CONCACAF", "Haiti":"CONCACAF", "Panama":"CONCACAF",
    # CAF (10)
    "Senegal":"CAF", "Morocco":"CAF", "Egypt":"CAF", "Algeria":"CAF",
    "Côte d'Ivoire":"CAF", "Cabo Verde":"CAF", "Ghana":"CAF", "Tunisia":"CAF",
    "South Africa":"CAF", "Congo DR":"CAF",
    # AFC (9)
    "Japan":"AFC", "Korea Republic":"AFC", "IR Iran":"AFC", "Jordan":"AFC",
    "Qatar":"AFC", "Uzbekistan":"AFC", "Saudi Arabia":"AFC", "Iraq":"AFC",
    "Australia":"AFC",
    # OFC (1)
    "New Zealand":"OFC",
}          # <- reuse the dictionary from Assessment 2 Task B

    base = add_form(build_base(path))

    m104 = build_104(base, rank, confed)
    m208 = build_208(base, rank, confed)

    print("matches :", len(m104), "rows   (need 104)")
    print("teams   :", len(m208), "rows   (need 208)")
    print("teams seen:", len(set(base.home_team) | set(base.away_team)), "(need 48)")

    m104.to_csv("matches_104.csv", index=False)
    m208.to_csv("teams_208.csv", index=False)
