# make_features_v2.py
# ------------------------------------------------------------
# Ajoute des features "propres" à fort levier :
# - Niveau tournoi (GS/M1000/ATP500/ATP250/Davis/Olympics) -> ordinal + one-hot
# - Main & Height via data/sackmann/atp_players.csv -> lefty flags + height diff
# - Forme par surface, H2H rate/conf (déjà), fatigue (short rest), transforms
# - Elo "récence" : décroissance vers 1500 (HL=180j) avant chaque match (global + surface)
# Sortie : work/outputs/features_v2.csv
# ------------------------------------------------------------

from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from unidecode import unidecode

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path("data")
SACK = DATA_DIR / "sackmann"
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
# Load matches
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
# Odds (pair le plus propre)
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

# proba implicite corrigée + odds_A/B
cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["odds_A"] = np.where(cond_AisW, df["oddsW"].to_numpy(), df["oddsL"].to_numpy())
df["odds_B"] = np.where(cond_AisW, df["oddsL"].to_numpy(), df["oddsW"].to_numpy())
df["p_book_A"] = (1.0 / df["odds_A"]) / ((1.0 / df["odds_A"]) + (1.0 / df["odds_B"]))

# ----------------------------
# Tourney level (GS/M1000/ATP500/ATP250/Davis/Olympics)
# ----------------------------
def infer_level_row(row) -> str:
    # 1) tennis-data 'Series' si présent
    if "Series" in row and isinstance(row["Series"], str) and row["Series"]:
        s = clean_text(row["Series"])
        if "grand slam" in s: return "GS"
        if "masters" in s or "1000" in s: return "M1000"
        if "500" in s: return "ATP500"
        if "250" in s: return "ATP250"
        if "olymp" in s: return "Olympics"
        if "davis" in s: return "Davis"
    # 2) sackmann 'tourney_level' si présent
    if "tourney_level" in row and isinstance(row["tourney_level"], str) and row["tourney_level"]:
        t = row["tourney_level"].upper()
        mapper = {"G":"GS","M":"M1000","A":"ATP500","B":"ATP250","D":"Davis","O":"Olympics","F":"Finals"}
        return mapper.get(t, t)
    # 3) heuristique sur noms
    name = ""
    for c in ["tourney_td","tourney_sack","Tournament"]:
        if c in row and isinstance(row[c], str) and row[c]:
            name = clean_text(row[c]); break
    if any(k in name for k in ["australian open","roland garros","french open","wimbledon","us open"]):
        return "GS"
    if any(k in name for k in ["indian wells","miami","monte carlo","madrid","rome","canada","cincinnati","shanghai","paris"]):
        return "M1000"
    if "olymp" in name: return "Olympics"
    if "davis" in name: return "Davis"
    return "ATP250"  # défaut conservateur

cand_cols = ["Series","tourney_level","tourney_td","tourney_sack","Tournament"]
for c in cand_cols:
    if c not in df.columns:
        df[c] = ""

df["tourney_level_norm"] = df.apply(infer_level_row, axis=1)
LEVEL_ORD = {"ATP250":1, "ATP500":2, "M1000":3, "GS":4, "Davis":3, "Olympics":3, "Finals":3}
df["level_ord"] = df["tourney_level_norm"].map(lambda x: LEVEL_ORD.get(x, 0)).astype(int)

for lv in ["ATP250","ATP500","M1000","GS","Davis","Olympics","Finals"]:
    df[f"level_{lv}"] = (df["tourney_level_norm"] == lv).astype(int)

# ----------------------------
# Elo (baseline) + Elo "récence" (HL=180j)
# ----------------------------
K_GLOBAL = 32.0
K_SURF   = 24.0
INIT_ELO = 1500.0
SURF_SET = {"hard","clay","grass","carpet"}
HALF_LIFE = 180.0  # jours
LAMBDA = np.log(2) / HALF_LIFE  # 1/HL ln2

df = df.sort_values(["match_date","level_ord","round_ord","A_key","B_key"]).reset_index(drop=True)

elo_g  = defaultdict(lambda: INIT_ELO)
elo_s  = {s: defaultdict(lambda: INIT_ELO) for s in SURF_SET}
last_upd_g = defaultdict(lambda: pd.NaT)
last_upd_s = {s: defaultdict(lambda: pd.NaT) for s in SURF_SET}

df["elo_w_pre"]  = np.nan; df["elo_l_pre"]  = np.nan
df["selo_w_pre"] = np.nan; df["selo_l_pre"] = np.nan
df["elo_w_decay_pre"]  = np.nan; df["elo_l_decay_pre"]  = np.nan
df["selo_w_decay_pre"] = np.nan; df["selo_l_decay_pre"] = np.nan

def expected(rA, rB): return 1.0 / (1.0 + 10.0 ** ((rB - rA)/400.0))

def decay_to_1500(rating, days):
    if pd.isna(days) or days <= 0:
        return rating
    # move delta towards 0 (1500 baseline)
    delta = rating - INIT_ELO
    factor = np.exp(-LAMBDA * days)
    return INIT_ELO + delta * factor

for i, r in df.iterrows():
    w = r["w_key"]; l = r["l_key"]
    d = r["match_date"]
    s = r["surface_final"] if r["surface_final"] in SURF_SET else "hard"

    # baseline Elo (sans décroissance) pré-match
    Ew_g, El_g = elo_g[w], elo_g[l]
    Ew_s, El_s = elo_s[s][w], elo_s[s][l]
    df.at[i, "elo_w_pre"]  = Ew_g; df.at[i, "elo_l_pre"]  = El_g
    df.at[i, "selo_w_pre"] = Ew_s; df.at[i, "selo_l_pre"] = El_s

    # Elo "décroissant" vers 1500 selon le temps depuis DERNIÈRE MAJ de ce joueur
    lwg, llg = last_upd_g[w], last_upd_g[l]
    lws, lls = last_upd_s[s][w], last_upd_s[s][l]

    days_w_g = (d - lwg).days if pd.notna(lwg) else np.nan
    days_l_g = (d - llg).days if pd.notna(llg) else np.nan
    days_w_s = (d - lws).days if pd.notna(lws) else np.nan
    days_l_s = (d - lls).days if pd.notna(lls) else np.nan

    Ew_g_dec = decay_to_1500(Ew_g, days_w_g)
    El_g_dec = decay_to_1500(El_g, days_l_g)
    Ew_s_dec = decay_to_1500(Ew_s, days_w_s)
    El_s_dec = decay_to_1500(El_s, days_l_s)

    df.at[i, "elo_w_decay_pre"]  = Ew_g_dec
    df.at[i, "elo_l_decay_pre"]  = El_g_dec
    df.at[i, "selo_w_decay_pre"] = Ew_s_dec
    df.at[i, "selo_l_decay_pre"] = El_s_dec

    # update baseline (pas la version décay) après le match courant (résultat : w gagne)
    pw  = expected(Ew_g, El_g)
    elo_g[w] = Ew_g + K_GLOBAL * (1.0 - pw)
    elo_g[l] = El_g + K_GLOBAL * (0.0 - (1.0 - pw))
    pws = expected(Ew_s, El_s)
    elo_s[s][w] = Ew_s + K_SURF * (1.0 - pws)
    elo_s[s][l] = El_s + K_SURF * (0.0 - (1.0 - pws))

    last_upd_g[w] = d; last_upd_g[l] = d
    last_upd_s[s][w] = d; last_upd_s[s][l] = d

# map vers A/B
cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["A_elo"]       = np.where(cond_AisW, df["elo_w_pre"].to_numpy(),        df["elo_l_pre"].to_numpy())
df["B_elo"]       = np.where(cond_AisW, df["elo_l_pre"].to_numpy(),        df["elo_w_pre"].to_numpy())
df["A_selo"]      = np.where(cond_AisW, df["selo_w_pre"].to_numpy(),       df["selo_l_pre"].to_numpy())
df["B_selo"]      = np.where(cond_AisW, df["selo_l_pre"].to_numpy(),       df["selo_w_pre"].to_numpy())
df["A_elo_decay"] = np.where(cond_AisW, df["elo_w_decay_pre"].to_numpy(),  df["elo_l_decay_pre"].to_numpy())
df["B_elo_decay"] = np.where(cond_AisW, df["elo_l_decay_pre"].to_numpy(),  df["elo_w_decay_pre"].to_numpy())
df["A_selo_decay"]= np.where(cond_AisW, df["selo_w_decay_pre"].to_numpy(), df["selo_l_decay_pre"].to_numpy())
df["B_selo_decay"]= np.where(cond_AisW, df["selo_l_decay_pre"].to_numpy(), df["selo_w_decay_pre"].to_numpy())

df["elo_diff"]        = df["A_elo"]       - df["B_elo"]
df["selo_diff"]       = df["A_selo"]      - df["B_selo"]
df["elo_decay_diff"]  = df["A_elo_decay"] - df["B_elo_decay"]
df["selo_decay_diff"] = df["A_selo_decay"]- df["B_selo_decay"]

# ----------------------------
# Forme globale & par surface (pré-match)
# ----------------------------
last_date   = defaultdict(lambda: pd.NaT)
recent_form = defaultdict(lambda: deque(maxlen=5))

df["rest_w_days"] = np.nan; df["rest_l_days"] = np.nan
df["form5_w"]     = np.nan; df["form5_l"]     = np.nan

# surface form
form_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))
df["form5surf_w"] = np.nan; df["form5surf_l"] = np.nan

for i, r in df.iterrows():
    w = r["w_key"]; l = r["l_key"]; d = r["match_date"]
    s = r["surface_final"] if r["surface_final"] in SURF_SET else "hard"

    lw = last_date[w]; ll = last_date[l]
    df.at[i, "rest_w_days"] = (d - lw).days if pd.notna(lw) else np.nan
    df.at[i, "rest_l_days"] = (d - ll).days if pd.notna(ll) else np.nan

    f_w = np.mean(recent_form[w]) if len(recent_form[w]) > 0 else np.nan
    f_l = np.mean(recent_form[l]) if len(recent_form[l]) > 0 else np.nan
    df.at[i, "form5_w"] = f_w; df.at[i, "form5_l"] = f_l

    fs_w = np.mean(form_surf[w][s]) if len(form_surf[w][s])>0 else np.nan
    fs_l = np.mean(form_surf[l][s]) if len(form_surf[l][s])>0 else np.nan
    df.at[i, "form5surf_w"] = fs_w; df.at[i, "form5surf_l"] = fs_l

    last_date[w] = d; last_date[l] = d
    recent_form[w].append(1.0); recent_form[l].append(0.0)
    form_surf[w][s].append(1.0); form_surf[l][s].append(0.0)

cond_AisW = eq_mask(df["A_key"], df["w_key"])
df["A_rest_days"] = np.where(cond_AisW, df["rest_w_days"].to_numpy(), df["rest_l_days"].to_numpy())
df["B_rest_days"] = np.where(cond_AisW, df["rest_l_days"].to_numpy(), df["rest_w_days"].to_numpy())
df["A_form5"]     = np.where(cond_AisW, df["form5_w"].to_numpy(),     df["form5_l"].to_numpy())
df["B_form5"]     = np.where(cond_AisW, df["form5_l"].to_numpy(),     df["form5_w"].to_numpy())
df["A_form5_surf"]= np.where(cond_AisW, df["form5surf_w"].to_numpy(), df["form5surf_l"].to_numpy())
df["B_form5_surf"]= np.where(cond_AisW, df["form5surf_l"].to_numpy(), df["form5surf_w"].to_numpy())

# ----------------------------
# H2H strict pré-match : rate & confidence
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
}).sort_values(["player","opponent","match_date","match_idx"]).reset_index(drop=True)

grp = long.groupby(["player","opponent"], sort=False)
long["h2h_played_pre"] = grp.cumcount()
long["h2h_wins_pre"]   = grp["is_win"].cumsum().shift(1, fill_value=0)

long_by_match = long.sort_values(["match_idx","is_win"], ascending=[True, False])
w_rows = long_by_match[long_by_match["is_win"] == 1]
l_rows = long_by_match[long_by_match["is_win"] == 0]
assert len(w_rows) == n and len(l_rows) == n

h2h_wins_w   = w_rows["h2h_wins_pre"].to_numpy()
h2h_played_w = w_rows["h2h_played_pre"].to_numpy()
h2h_wins_l   = l_rows["h2h_wins_pre"].to_numpy()
h2h_played_l = l_rows["h2h_played_pre"].to_numpy()

df["A_h2h_wins"]   = np.where(cond_AisW, h2h_wins_w,   h2h_wins_l)
df["A_h2h_played"] = np.where(cond_AisW, h2h_played_w, h2h_played_l)
df["B_h2h_wins"]   = np.where(cond_AisW, h2h_wins_l,   h2h_wins_w)
df["B_h2h_played"] = np.where(cond_AisW, h2h_played_l, h2h_played_w)

def _safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce").clip(lower=1)
    return (num/den).astype(float)

for side in ["A","B"]:
    df[f"{side}_h2h_rate"] = _safe_div(df[f"{side}_h2h_wins"], df[f"{side}_h2h_played"])
    df[f"{side}_h2h_conf"] = 1.0 / np.sqrt(pd.to_numeric(df[f"{side}_h2h_played"], errors="coerce").clip(lower=1))

df["h2h_diff"]       = (pd.to_numeric(df["A_h2h_wins"], errors="coerce") - pd.to_numeric(df["B_h2h_wins"], errors="coerce")).astype(float)
df["h2h_rate_diff"]  = df["A_h2h_rate"] - df["B_h2h_rate"]
df["h2h_conf_diff"]  = df["A_h2h_conf"] - df["B_h2h_conf"]

# ----------------------------
# Fatigue / transforms
# ----------------------------
def _winsor_clip(x, lo=0, hi=14):
    x = pd.to_numeric(x, errors="coerce")
    return x.clip(lower=lo, upper=hi)

df["A_rest_log"] = np.log1p(_winsor_clip(df["A_rest_days"]))
df["B_rest_log"] = np.log1p(_winsor_clip(df["B_rest_days"]))
df["rest_log_diff"] = df["A_rest_log"] - df["B_rest_log"]
df["A_short_rest_lt3"] = (pd.to_numeric(df["A_rest_days"], errors="coerce") < 3).astype(int)
df["B_short_rest_lt3"] = (pd.to_numeric(df["B_rest_days"], errors="coerce") < 3).astype(int)

# ----------------------------
# Main & Height depuis atp_players.csv
# ----------------------------
def load_players_map(players_csv: Path):
    if not players_csv.exists():
        return {}
    pl = pd.read_csv(players_csv, dtype=str, low_memory=False)
    first = pl.get("name_first", pd.Series([], dtype="string")).fillna("").apply(clean_text)
    last  = pl.get("name_last",  pd.Series([], dtype="string")).fillna("").apply(clean_text)
    hand  = pl.get("hand",       pd.Series([], dtype="string")).fillna("").str.upper()
    hcm   = pd.to_numeric(pl.get("height_cm", pd.Series([])), errors="coerce")

    out = {}
    for f, l, h, cm in zip(first, last, hand, hcm):
        key = player_key_sack((f + " " + l).strip())
        out[key] = {
            "lefty": 1 if (isinstance(h, str) and h.startswith("L")) else 0,
            "height_cm": float(cm) if pd.notna(cm) else np.nan
        }
    return out

players_map = load_players_map(SACK / "atp_players.csv")

def map_attr_from_players(keys: pd.Series, attr: str):
    vals = []
    for k in keys.fillna(""):
        v = players_map.get(str(k), None)
        if v is None: vals.append(np.nan)
        else: vals.append(v.get(attr, np.nan))
    return np.array(vals, dtype=float)

df["A_lefty"]      = np.where(cond_AisW, map_attr_from_players(df["w_key"], "lefty"), map_attr_from_players(df["l_key"], "lefty"))
df["B_lefty"]      = np.where(cond_AisW, map_attr_from_players(df["l_key"], "lefty"), map_attr_from_players(df["w_key"], "lefty"))
df["A_height_cm"]  = np.where(cond_AisW, map_attr_from_players(df["w_key"], "height_cm"), map_attr_from_players(df["l_key"], "height_cm"))
df["B_height_cm"]  = np.where(cond_AisW, map_attr_from_players(df["l_key"], "height_cm"), map_attr_from_players(df["w_key"], "height_cm"))
df["height_diff"]  = df["A_height_cm"] - df["B_height_cm"]
df["lefty_vs_righty"] = ( (df["A_lefty"]==1) & (df["B_lefty"]==0) ).astype(int)

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
cols_basic = ["round_ord","best_of","surf_hard","surf_clay","surf_grass","surf_carpet",
              "tourney_level_norm","level_ord","level_ATP250","level_ATP500","level_M1000","level_GS","level_Davis","level_Olympics","level_Finals"]
cols_rank_age = []  # si tu ajoutes plus tard w_rank/l_rank et ages
cols_elo = ["A_elo","B_elo","elo_diff","A_selo","B_selo","selo_diff",
            "A_elo_decay","B_elo_decay","elo_decay_diff","A_selo_decay","B_selo_decay","selo_decay_diff"]
cols_form_rest = ["A_form5","B_form5","A_rest_days","B_rest_days","A_form5_surf","B_form5_surf","rest_log_diff","A_short_rest_lt3","B_short_rest_lt3"]
cols_h2h = ["A_h2h_wins","A_h2h_played","B_h2h_wins","B_h2h_played","h2h_diff",
            "A_h2h_rate","B_h2h_rate","h2h_rate_diff","A_h2h_conf","B_h2h_conf","h2h_conf_diff"]
cols_players = ["A_lefty","B_lefty","lefty_vs_righty","A_height_cm","B_height_cm","height_diff"]

keep = cols_id + cols_odds + cols_target + cols_basic + cols_rank_age + cols_elo + cols_form_rest + cols_h2h + cols_players
keep = [c for c in keep if c in df.columns]
features = df[keep].copy()

out_path = OUTDIR / "features_v2.csv"
features.to_csv(out_path, index=False)
print(f"OK - features écrites : {out_path}")
print("Lignes :", len(features))
print("Cible non nulle (y):", features["y"].notna().sum())
print("Couverture book (p_book_A non nulle):", features["p_book_A"].notna().sum())
