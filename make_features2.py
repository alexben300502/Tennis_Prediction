# make_features2.py
# ------------------------------------------------------------
# Dataset "pré-match" sans fuite :
# - A/B via favori marché (PS -> Avg -> B365 -> autres)
# - Elo global + par surface (pré-match) via boucle séquentielle
# - H2H pré-match, Forme 5, Repos (jours)
# - p_book_A (proba implicite corrigée de l'overround)
# Patches:
#   * Comparaisons A_key==w_key via eq_mask(...) (pas de pd.NA en bool)
#   * Construction H2H "long" en 2*n (pas de mismatch de shapes)
#   * best_of géré sans .fillna sur str
#   * y figé en numpy int8 (jamais nullable)
# Sortie : work/outputs/features_v1.csv
# ------------------------------------------------------------

from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from unidecode import unidecode

# ----------------------------
# Chemins
# ----------------------------
OUTDIR = Path("work/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
SRC1 = OUTDIR / "matches_final2.csv"
SRC0 = OUTDIR / "matches_final.csv"
SRC  = SRC1 if SRC1.exists() else SRC0
if not SRC.exists():
    raise FileNotFoundError("Aucun matches_final*.csv trouvé dans work/outputs/")

# ----------------------------
# Helpers
# ----------------------------
def ensure_str_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.Series(df[col], index=df.index, dtype="string").fillna("")
    else:
        df[col] = pd.Series([""] * len(df), index=df.index, dtype="string")

def clean_text(s: str) -> str:
    if pd.isna(s): return ""
    s = unidecode(str(s)).lower().strip()
    for ch in [".", ",", "'", '"', "(", ")", "-", "–", "&", "/"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())

SURFACE_MAP = {
    "hard":"hard","h":"hard","i hard":"hard","indoor hard":"hard",
    "clay":"clay","c":"clay","red clay":"clay",
    "grass":"grass","g":"grass",
    "carpet":"carpet","i carpet":"carpet","indoor carpet":"carpet"
}
def canonical_surface(s: str) -> str:
    s0 = clean_text(s)
    return SURFACE_MAP.get(s0, s0)

ROUND_CANON = {
    "r128":"R128","r64":"R64","r32":"R32","r16":"R16","qf":"QF","sf":"SF","f":"F",
    "first round":"R64","1st round":"R64","second round":"R32","2nd round":"R32",
    "third round":"R16","3rd round":"R16","round of 64":"R64","round of 32":"R32","round of 16":"R16",
    "quarterfinal":"QF","quarterfinals":"QF","quarter-finals":"QF",
    "semifinal":"SF","semi-final":"SF","semifinals":"SF","final":"F","finals":"F"
}
ROUND_ORD = {"R128":1,"R64":2,"R32":3,"R16":4,"QF":5,"SF":6,"F":7}

def canonical_round(r: str) -> str:
    r0 = clean_text(r)
    return ROUND_CANON.get(r0, r0.upper())

def parse_date(sr) -> pd.Series:
    out = pd.to_datetime(sr, errors="coerce")
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask].astype(str), format="%Y%m%d", errors="coerce")
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask], format="%d/%m/%Y", errors="coerce")
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask], format="%m/%d/%Y", errors="coerce")
    return out

# Comparaison sûre (évite pd.NA en bool)
def eq_mask(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a.fillna("").astype(str).to_numpy() == b.fillna("").astype(str).to_numpy())

# Keys joueurs
def player_key_td(name: str) -> str:
    s = clean_text(name)
    if not s: return s
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    k = 0
    for t in reversed(parts):
        if len(t) == 1:
            k += 1
        else:
            break
    if k >= 1:
        last = " ".join(parts[:-k])
        ini = parts[-k]
        return f"{ini} {last}".strip()
    ini = parts[0][0]
    last = " ".join(parts[1:])
    return f"{ini} {last}".strip()

def player_key_sack(name: str) -> str:
    s = clean_text(name)
    if not s: return s
    parts = s.split()
    ini = parts[0][0] if parts else ""
    last = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{ini} {last}".strip()

# ----------------------------
# Load & colonnes de base
# ----------------------------
df = pd.read_csv(SRC, low_memory=False)

date_cols = [c for c in ["match_date_td","tourney_start_date","date","match_date"] if c in df.columns]
if not date_cols:
    raise RuntimeError("Aucune colonne de date trouvée dans matches_final*.csv")
df["match_date"] = parse_date(df[date_cols[0]])

surf_col = "surface_td" if "surface_td" in df.columns else "surface_sack"
ensure_str_col(df, surf_col)
df["surface_final"] = df[surf_col].map(canonical_surface)

round_col = "round_td" if "round_td" in df.columns else "round_sack"
ensure_str_col(df, round_col)
df["round_final"] = df[round_col].map(canonical_round)
df["round_ord"]   = df["round_final"].map(lambda x: ROUND_ORD.get(x, 0)).astype(int)

# Winner/Loser keys
if {"winner_td","loser_td"} <= set(df.columns):
    w_name, l_name, key_fn = "winner_td", "loser_td", player_key_td
elif {"winner_name","loser_name"} <= set(df.columns):
    w_name, l_name, key_fn = "winner_name", "loser_name", player_key_sack
else:
    for a, b in [("Winner","Loser"),("winner","loser")]:
        if {a, b} <= set(df.columns):
            w_name, l_name, key_fn = a, b, player_key_td
            break
    else:
        raise RuntimeError("Impossible d'identifier Winner/Loser.")

ensure_str_col(df, w_name); ensure_str_col(df, l_name)
df["w_key"] = df[w_name].map(key_fn)
df["l_key"] = df[l_name].map(key_fn)

# ----------------------------
# Cotes : meilleur pair dispo
# ----------------------------
BOOK_PAIRS = [
    ("PSW","PSL"), ("AvgW","AvgL"), ("B365W","B365L"), ("EXW","EXL"), ("IWW","IWL")
]
oddsW = pd.Series(np.nan, index=df.index, dtype="float64")
oddsL = pd.Series(np.nan, index=df.index, dtype="float64")
book_used = pd.Series("", index=df.index, dtype="string")

for wcol, lcol in BOOK_PAIRS:
    if wcol in df.columns and lcol in df.columns:
        m = oddsW.isna() & df[wcol].notna() & df[lcol].notna()
        oddsW.loc[m] = pd.to_numeric(df.loc[m, wcol], errors="coerce")
        oddsL.loc[m] = pd.to_numeric(df.loc[m, lcol], errors="coerce")
        book_used.loc[m] = wcol[:-1]

df["oddsW"] = oddsW
df["oddsL"] = oddsL
df["book_used"] = book_used
df = df[df["oddsW"].notna() & df["oddsL"].notna()].copy()

# ----------------------------
# Orientation A/B via cotes
# ----------------------------
A_key, B_key, y = [], [], []
for _, r in df.iterrows():
    w, l = r["w_key"], r["l_key"]
    ow, ol = float(r["oddsW"]), float(r["oddsL"])
    if ow < ol:
        A_key.append(w); B_key.append(l); y.append(1)
    elif ow > ol:
        A_key.append(l); B_key.append(w); y.append(0)
    else:
        a, b = sorted([w, l])
        A_key.append(a); B_key.append(b); y.append(1 if w == a else 0)

df["A_key"] = pd.Series(A_key, index=df.index, dtype="string")
df["B_key"] = pd.Series(B_key, index=df.index, dtype="string")
df["y"]     = pd.Series(np.array(y, dtype=np.int8), index=df.index)
assert df["y"].notna().all(), "y contient des NaN après orientation A/B"

# p_book_A (corrigé de l'overround) + odds_A/B
cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["odds_A"] = np.where(cond_AisW, df["oddsW"].to_numpy(), df["oddsL"].to_numpy())
df["odds_B"] = np.where(cond_AisW, df["oddsL"].to_numpy(), df["oddsW"].to_numpy())
df["p_book_A"] = (1.0 / df["odds_A"]) / ((1.0 / df["odds_A"]) + (1.0 / df["odds_B"]))

# ----------------------------
# Elo pré-match (global + surface) sans fuite
# ----------------------------
K_GLOBAL = 32.0
K_SURF   = 24.0
INIT_ELO = 1500.0
SURF_SET = {"hard","clay","grass","carpet"}

df = df.sort_values(["match_date","round_ord","A_key","B_key"]).reset_index(drop=True)

elo_g  = defaultdict(lambda: INIT_ELO)
elo_s  = {s: defaultdict(lambda: INIT_ELO) for s in SURF_SET}

df["elo_w_pre"]  = np.nan
df["elo_l_pre"]  = np.nan
df["selo_w_pre"] = np.nan
df["selo_l_pre"] = np.nan

def expected(rA, rB):
    return 1.0 / (1.0 + 10.0 ** ((rB - rA) / 400.0))

for i, r in df.iterrows():
    w = r["w_key"]; l = r["l_key"]
    s = r["surface_final"] if r["surface_final"] in SURF_SET else "hard"

    Ew_g, El_g = elo_g[w], elo_g[l]
    Ew_s, El_s = elo_s[s][w], elo_s[s][l]

    df.at[i, "elo_w_pre"]  = Ew_g
    df.at[i, "elo_l_pre"]  = El_g
    df.at[i, "selo_w_pre"] = Ew_s
    df.at[i, "selo_l_pre"] = El_s

    pw  = expected(Ew_g, El_g)
    elo_g[w] = Ew_g + K_GLOBAL * (1.0 - pw)
    elo_g[l] = El_g + K_GLOBAL * (0.0 - (1.0 - pw))

    pws = expected(Ew_s, El_s)
    elo_s[s][w] = Ew_s + K_SURF * (1.0 - pws)
    elo_s[s][l] = El_s + K_SURF * (0.0 - (1.0 - pws))

# map vers A/B
cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["A_elo"]  = np.where(cond_AisW, df["elo_w_pre"].to_numpy(),  df["elo_l_pre"].to_numpy())
df["B_elo"]  = np.where(cond_AisW, df["elo_l_pre"].to_numpy(),  df["elo_w_pre"].to_numpy())
df["A_selo"] = np.where(cond_AisW, df["selo_w_pre"].to_numpy(), df["selo_l_pre"].to_numpy())
df["B_selo"] = np.where(cond_AisW, df["selo_l_pre"].to_numpy(), df["selo_w_pre"].to_numpy())

df["elo_diff"]  = df["A_elo"]  - df["B_elo"]
df["selo_diff"] = df["A_selo"] - df["B_selo"]

# ----------------------------
# Repos (jours) & Forme 5 (pré-match)
# ----------------------------
last_date   = defaultdict(lambda: pd.NaT)
recent_form = defaultdict(lambda: deque(maxlen=5))

df["rest_w_days"] = np.nan
df["rest_l_days"] = np.nan
df["form5_w"]     = np.nan
df["form5_l"]     = np.nan

for i, r in df.iterrows():
    w = r["w_key"]; l = r["l_key"]; d = r["match_date"]

    lw = last_date[w]; ll = last_date[l]
    df.at[i, "rest_w_days"] = (d - lw).days if pd.notna(lw) else np.nan
    df.at[i, "rest_l_days"] = (d - ll).days if pd.notna(ll) else np.nan

    f_w = np.mean(recent_form[w]) if len(recent_form[w]) > 0 else np.nan
    f_l = np.mean(recent_form[l]) if len(recent_form[l]) > 0 else np.nan
    df.at[i, "form5_w"] = f_w
    df.at[i, "form5_l"] = f_l

    last_date[w] = d; last_date[l] = d
    recent_form[w].append(1.0); recent_form[l].append(0.0)

cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["A_rest_days"] = np.where(cond_AisW, df["rest_w_days"].to_numpy(), df["rest_l_days"].to_numpy())
df["B_rest_days"] = np.where(cond_AisW, df["rest_l_days"].to_numpy(), df["rest_w_days"].to_numpy())
df["A_form5"]     = np.where(cond_AisW, df["form5_w"].to_numpy(),     df["form5_l"].to_numpy())
df["B_form5"]     = np.where(cond_AisW, df["form5_l"].to_numpy(),     df["form5_w"].to_numpy())

# ----------------------------
# H2H strict pré-match (construction 2*n)
# ----------------------------
n = len(df)
match_idx = np.repeat(df.index.values, 2)

player   = np.empty(2*n, dtype=object)
opponent = np.empty(2*n, dtype=object)
is_win   = np.empty(2*n, dtype=np.int8)
mdate    = np.empty(2*n, dtype="datetime64[ns]")

player[0::2]   = df["w_key"].to_numpy()
player[1::2]   = df["l_key"].to_numpy()
opponent[0::2] = df["l_key"].to_numpy()
opponent[1::2] = df["w_key"].to_numpy()
is_win[0::2]   = 1
is_win[1::2]   = 0
mdate[0::2]    = df["match_date"].to_numpy()
mdate[1::2]    = df["match_date"].to_numpy()

long = pd.DataFrame({
    "match_idx": match_idx,
    "player":    player,
    "opponent":  opponent,
    "is_win":    is_win,
    "match_date": mdate
})
long = long.sort_values(["player","opponent","match_date","match_idx"]).reset_index(drop=True)

grp = long.groupby(["player","opponent"], sort=False)
long["h2h_played_pre"] = grp.cumcount()
long["h2h_wins_pre"]   = grp["is_win"].cumsum().shift(1, fill_value=0)

# Récupérer par match : winner d'abord (is_win=1), loser ensuite
long_by_match = long.sort_values(["match_idx","is_win"], ascending=[True, False])
w_rows = long_by_match[long_by_match["is_win"] == 1]
l_rows = long_by_match[long_by_match["is_win"] == 0]
assert len(w_rows) == n and len(l_rows) == n, "Incohérence H2H: w_rows/l_rows."

h2h_wins_w   = w_rows["h2h_wins_pre"].to_numpy()
h2h_played_w = w_rows["h2h_played_pre"].to_numpy()
h2h_wins_l   = l_rows["h2h_wins_pre"].to_numpy()
h2h_played_l = l_rows["h2h_played_pre"].to_numpy()

cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["A_h2h_wins"]   = np.where(cond_AisW, h2h_wins_w,   h2h_wins_l)
df["A_h2h_played"] = np.where(cond_AisW, h2h_played_w, h2h_played_l)
df["B_h2h_wins"]   = np.where(cond_AisW, h2h_wins_l,   h2h_wins_w)
df["B_h2h_played"] = np.where(cond_AisW, h2h_played_l, h2h_played_w)

df["h2h_diff"] = (pd.to_numeric(df["A_h2h_wins"], errors="coerce")
                  - pd.to_numeric(df["B_h2h_wins"], errors="coerce")).astype(float)

# ----------------------------
# Rank / Age si dispo
# ----------------------------
def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

if {"w_rank","l_rank"} <= set(df.columns):
    df["A_rank"] = np.where(cond_AisW, safe_num(df["w_rank"]).to_numpy(), safe_num(df["l_rank"]).to_numpy())
    df["B_rank"] = np.where(cond_AisW, safe_num(df["l_rank"]).to_numpy(), safe_num(df["w_rank"]).to_numpy())
    df["rank_diff"] = (df["B_rank"] - df["A_rank"]).astype(float)

if {"winner_age","loser_age"} <= set(df.columns):
    df["A_age"] = np.where(cond_AisW, safe_num(df["winner_age"]).to_numpy(), safe_num(df["loser_age"]).to_numpy())
    df["B_age"] = np.where(cond_AisW, safe_num(df["loser_age"]).to_numpy(), safe_num(df["winner_age"]).to_numpy())
    df["age_diff"] = (df["A_age"] - df["B_age"]).astype(float)

# ----------------------------
# One-hot surface & best_of
# ----------------------------
for s in ["hard","clay","grass","carpet"]:
    df[f"surf_{s}"] = (df["surface_final"] == s).astype(int)

if "best_of" in df.columns:
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
else:
    df["best_of"] = 3

# ----------------------------
# Sélection & export
# ----------------------------
cols_id = ["match_date","A_key","B_key","book_used"]
cols_odds = ["odds_A","odds_B","p_book_A"]
cols_target = ["y"]
cols_basic = ["round_ord","best_of","surf_hard","surf_clay","surf_grass","surf_carpet"]
cols_rank_age = [c for c in ["A_rank","B_rank","rank_diff","A_age","B_age","age_diff"] if c in df.columns]
cols_elo = ["A_elo","B_elo","elo_diff","A_selo","B_selo","selo_diff"]
cols_form_rest = ["A_form5","B_form5","A_rest_days","B_rest_days"]
cols_h2h = ["A_h2h_wins","A_h2h_played","B_h2h_wins","B_h2h_played","h2h_diff"]

keep = cols_id + cols_odds + cols_target + cols_basic + cols_rank_age + cols_elo + cols_form_rest + cols_h2h
keep = [c for c in keep if c in df.columns]
features = df[keep].copy()

# sécurité finale sur y (numpy int8 et pas de NaN)
if features["y"].isna().any():
    print("⚠️  y contient des NaN, correction forcée (mise à 0).")
    print(features.loc[features["y"].isna(), ["A_key","B_key","odds_A","odds_B"]].head())
    features["y"] = features["y"].fillna(0)
features["y"] = np.array(features["y"], dtype=np.int8)

# casts num utiles
for c in ["A_form5","B_form5","A_rest_days","B_rest_days","A_h2h_wins","A_h2h_played","B_h2h_wins","B_h2h_played"]:
    if c in features.columns:
        features[c] = pd.to_numeric(features[c], errors="coerce")

out_path = OUTDIR / "features_v1.csv"
features.to_csv(out_path, index=False)
print(f"OK - features écrites : {out_path}")
print("Lignes :", len(features))
print("Cible non nulle (y):", features["y"].notna().sum())
print("Couverture book (p_book_A non nulle):", features["p_book_A"].notna().sum())
