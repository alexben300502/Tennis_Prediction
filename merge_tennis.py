# merge_tennis.py
# ------------------------------------------------------------
# Fusionne Jeff Sackmann (résultats/stats) et tennis-data (odds)
# Clés robustes:
#  - tournoi = VILLE + SURFACE (Sackmann: tourney_name ~ ville ; tennis-data: Location, sinon Tournament)
#  - joueurs = "initiale + nom"
#  - pas de jointure sur la date; fallback: on choisit la date TD la plus proche (±21 j) en préférant le même round
# Correctifs:
#  - pas de .fillna() sur str (ensure_str_col), parsing dates vectorisé (parse_date_series)
#  - détection nette des colonnes d'odds (regex)
#  - tkey_city_surface vectorisé (tkey_city_surface_vec)
#  - fallback supplémentaire "ville seule" ±14 j (sans surface)
#  - export de colonnes Sackmann utiles (winner_*, loser_*, w_*, l_*, best_of, minutes)
# ------------------------------------------------------------

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from unidecode import unidecode  # pip install Unidecode

# ------------------------
# Config
# ------------------------
DATA_DIR = Path("data")
SACK = DATA_DIR / "sackmann"
TD = DATA_DIR / "tennisdata"
WORK = Path("work")
CACHE = WORK / "cache"
OUT = WORK / "outputs"
for p in (CACHE, OUT):
    p.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2017, 2026))  # 2017-2025 inclus
DATE_WINDOW_DAYS = 21            # fenêtre pour fallback par date (ville+surface)
DATE_WINDOW_CITY_ONLY = 14       # fenêtre pour fallback par date (ville seule)
MAIN_ROUNDS = {"R128","R64","R32","R16","QF","SF","F"}

# Aliases de villes/lieux pour tennis-data / Sackmann -> aide à matcher
TOURNEY_ALIASES = {
    # Paris
    "paris bercy": "paris", "bercy": "paris", "roland garros": "paris",
    # Indian Wells / Miami
    "indian wells": "indian wells", "miami gardens": "miami", "miami": "miami",
    # Masters / ATP 500 / 250 courants
    "monte carlo": "monte carlo",
    "madrid": "madrid",
    "rome": "rome", "roma": "rome",
    "rotterdam": "rotterdam", "abn amro": "rotterdam",
    "doha": "doha", "qatar": "doha",
    "dubai": "dubai", "dubai duty free": "dubai",
    "acapulco": "acapulco", "mexico": "acapulco",
    "washington": "washington", "citi open": "washington",
    "metz": "metz", "moselle": "metz",
    "st petersburg": "saint petersburg", "saint petersburg": "saint petersburg",
    "kitzbuhel": "kitzbuehel", "kitzbuehel": "kitzbuehel",
    "s hertogenbosch": "den bosch", "s-hertogenbosch": "den bosch", "den bosch": "den bosch", "hertogenbosch": "den bosch",
    "antwerp": "antwerp", "antwerpen": "antwerp",
    "sydney": "sydney",
    "adelaide": "adelaide",
    "auckland": "auckland",
    "sao paulo": "sao paulo", "são paulo": "sao paulo",
    "rio de janeiro": "rio de janeiro", "rio": "rio de janeiro",
    "houston": "houston",
    "buenos aires": "buenos aires",
    "barcelona": "barcelona",
    "vienna": "vienna", "wien": "vienna",
    "stuttgart": "stuttgart",
    "shanghai": "shanghai",
    "tokyo": "tokyo",
    "beijing": "beijing",
    # Canada alternance : on garde chaque ville telle quelle
    "toronto": "toronto", "montreal": "montreal",
    "cincinnati": "cincinnati",
    "london": "london", "londres": "london",
    "metz": "metz", "moselle": "metz", 
    "australian open": "melbourne",
    "wimbledon": "london",
    "us open": "new york", "us open ": "new york", "us  open": "new york",  # variations fréquentes
    "indian wells masters": "indian wells",
    "miami masters": "miami",
    "rome masters": "rome",
    "madrid masters": "madrid",
    "cincinnati masters": "cincinnati",
    "paris masters": "paris",
    "monte carlo masters": "monte carlo",
    "queen s club": "london", "queens club": "london",
    "shanghai masters": "shanghai",
    "great ocean road open": "melbourne",
    "adelaide 1": "adelaide",
    "astana": "astana", "nur sultan": "astana"
}

# ------------------------
# Helpers
# ------------------------
def ensure_str_col(df: pd.DataFrame, col: str) -> None:
    """Garanti df[col] en dtype string pandas, sans NaN."""
    if col in df.columns:
        df[col] = pd.Series(df[col], index=df.index, dtype="string").fillna("")
    else:
        df[col] = pd.Series([""] * len(df), index=df.index, dtype="string")

def clean_text(s: str) -> str:
    if pd.isna(s): return ""
    s = unidecode(str(s)).lower().strip()
    for ch in [".", ",", "'", '"', "(", ")", "-", "–", "&", "/"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

SURFACE_MAP = {
    "hard":"hard", "h":"hard", "i hard":"hard", "indoor hard":"hard",
    "clay":"clay","c":"clay","red clay":"clay",
    "grass":"grass","g":"grass",
    "carpet":"carpet","i carpet":"carpet","indoor carpet":"carpet"
}
def canonical_surface(s: str) -> str:
    return SURFACE_MAP.get(clean_text(s), clean_text(s))

ROUND_MAP = {
    "first round":"R64","1st round":"R64",
    "second round":"R32","2nd round":"R32",
    "third round":"R16","3rd round":"R16",
    "round of 64":"R64","round of 32":"R32","round of 16":"R16",
    "quarterfinal":"QF","quarter-finals":"QF","quarterfinals":"QF",
    "semifinal":"SF","semi-final":"SF","semifinals":"SF",
    "final":"F","finals":"F"
}
def canonical_round(r: str) -> str:
    r0 = clean_text(r)
    if r0 in ROUND_MAP:
        return ROUND_MAP[r0]
    if r0 in {"r128","r64","r32","r16","qf","sf","f"}:
        return r0.upper()
    return r0.upper()

def alias_city(city: str) -> str:
    c = clean_text(city)
    return TOURNEY_ALIASES.get(c, c)

def tkey_city_surface_vec(city_sr: pd.Series, surface_sr: pd.Series) -> pd.Series:
    """Version vectorisée: (ville aliasée + surface canonique) pour chaque ligne."""
    city_clean = city_sr.fillna("").astype(str).map(alias_city)
    surf_clean = surface_sr.fillna("").astype(str).map(canonical_surface)
    out = (city_clean + "_" + surf_clean).str.strip("_")
    return out

def city_key_vec(city_sr: pd.Series) -> pd.Series:
    """Clé ville seule (aliasée)."""
    return city_sr.fillna("").astype(str).map(alias_city)

# --- Joueurs -> "initiale + nom" ---
def build_player_key_map(players_csv: Path) -> dict:
    """Construit un dict 'first last' -> 'f last' depuis atp_players.csv (Sackmann)."""
    if not players_csv.exists():
        return {}
    # DtypeWarning -> dtype=str et low_memory=False
    players = pd.read_csv(players_csv, dtype=str, low_memory=False)
    first = players.get("name_first", pd.Series([], dtype="string")).fillna("").apply(clean_text)
    last  = players.get("name_last",  pd.Series([], dtype="string")).fillna("").apply(clean_text)
    m = {}
    for f, l in zip(first, last):
        if l:
            full = (f + " " + l).strip()
            ini = f[0] if f else ""
            m[clean_text(full)] = f"{ini} {l}".strip()
    return m

PLAYER_KEY_MAP = build_player_key_map(SACK / "atp_players.csv")

def player_key_sack(name: str) -> str:
    """Sackmann -> map 'first last' -> 'f last' si connu, sinon heuristique."""
    s = clean_text(name)
    if not s: return s
    if s in PLAYER_KEY_MAP:
        return PLAYER_KEY_MAP[s]
    parts = s.split()
    ini = parts[0][0] if parts else ""
    last = " ".join(parts[1:]) if len(parts) > 1 else ""
    return f"{ini} {last}".strip()

def player_key_td(name: str) -> str:
    """Tennis-data -> 'Last F.'/'Last F.M.' -> 'f last'; sinon fallback."""
    s = clean_text(name)
    if not s: return s
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    # remonter les initiales finales
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
    return player_key_sack(s)

def parse_date_series(sr: pd.Series) -> pd.Series:
    out = pd.to_datetime(sr, errors="coerce")  # auto
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask].astype(str), format="%Y%m%d", errors="coerce")
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask], format="%d/%m/%Y", errors="coerce")
    mask = out.isna()
    out.loc[mask] = pd.to_datetime(sr[mask], format="%m/%d/%Y", errors="coerce")
    return out

# ------------------------
# Chargeurs
# ------------------------
def load_sackmann(years):
    frames = []
    for y in years:
        f = SACK / f"atp_matches_{y}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)

        # date = début de tournoi (YYYYMMDD)
        if "tourney_date" in df.columns:
            df["date_tourney"] = parse_date_series(df["tourney_date"].astype(str))
        elif "date" in df.columns:
            df["date_tourney"] = parse_date_series(df["date"])
        else:
            df["date_tourney"] = pd.NaT

        for col in ["tourney_name","surface","round","winner_name","loser_name","score"]:
            ensure_str_col(df, col)

        df["surface_c"] = df["surface"].apply(canonical_surface)
        df["rnd_canon"] = df["round"].apply(canonical_round)

        # clés tournoi
        df["tkey_city"] = tkey_city_surface_vec(df["tourney_name"], df["surface_c"])
        df["city_key"]  = city_key_vec(df["tourney_name"])

        # joueurs -> "f last"
        df["w_key"] = df["winner_name"].apply(player_key_sack)
        df["l_key"] = df["loser_name"].apply(player_key_sack)
        df["players_sorted"] = df[["w_key","l_key"]].apply(lambda x: "__".join(sorted(x.astype(str).tolist())), axis=1)

        # clés (sans date)
        df["key_city_players"]         = df["tkey_city"] + "__" + df["players_sorted"]
        df["key_city_round_players"]   = df["tkey_city"] + "__" + df["rnd_canon"] + "__" + df["players_sorted"]
        df["key_cityonly_players"]     = df["city_key"]  + "__" + df["players_sorted"]
        df["key_cityonly_round_players"]= df["city_key"] + "__" + df["rnd_canon"] + "__" + df["players_sorted"]

        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

TD_RENAME = {
    "date":"Date","tournament":"Tournament","tourney":"Tournament","event":"Tournament",
    "surface":"Surface","round":"Round","rnd":"Round",
    "winner":"Winner","player 1":"Winner","p1":"Winner",
    "loser":"Loser","player 2":"Loser","p2":"Loser",
    "location":"Location"
}
def load_tennisdata_year(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p, engine="openpyxl")
    elif p.suffix.lower() == ".xls":
        df = pd.read_excel(p, engine="xlrd")
    else:
        df = pd.read_csv(p)

    # rename souple
    ren = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in TD_RENAME:
            ren[c] = TD_RENAME[lc]
    df = df.rename(columns=ren)

    # colonnes utiles
    for col in ["Date","Tournament","Location","Surface","Round","Winner","Loser"]:
        ensure_str_col(df, col)

    df["Date_dt"]   = parse_date_series(df["Date"])
    df["Surface_c"] = df["Surface"].apply(canonical_surface)
    df["rnd_canon"] = df["Round"].apply(canonical_round)

    # city pour clé: Location si dispo sinon Tournament
    city_for_key = df["Location"]
    missing_city = city_for_key.str.len() == 0
    if missing_city.any():
        city_for_key = city_for_key.mask(missing_city, df["Tournament"])

    df["tkey_city"] = tkey_city_surface_vec(city_for_key, df["Surface_c"])
    df["city_key"]  = city_key_vec(city_for_key)

    # joueurs "f last"
    df["w_key"] = df["Winner"].apply(player_key_td)
    df["l_key"] = df["Loser"].apply(player_key_td)
    df["players_sorted"] = df[["w_key","l_key"]].apply(lambda x: "__".join(sorted(x.astype(str).tolist())), axis=1)

    # clés
    df["key_city_players"]          = df["tkey_city"] + "__" + df["players_sorted"]
    df["key_city_round_players"]    = df["tkey_city"] + "__" + df["rnd_canon"] + "__" + df["players_sorted"]
    df["key_cityonly_players"]      = df["city_key"]  + "__" + df["players_sorted"]
    df["key_cityonly_round_players"]= df["city_key"]  + "__" + df["rnd_canon"] + "__" + df["players_sorted"]
    return df

def load_tennisdata(years):
    frames = []
    for y in years:
        candidates = [
            TD / f"{y}.xlsx", TD / f"{y}.xls", TD / f"{y}.csv",
            TD / f"tennis_{y}.xlsx", TD / f"tennis_{y}.csv"
        ]
        path = next((p for p in candidates if p.exists()), None)
        if not path:
            continue
        dfy = load_tennisdata_year(path)
        dfy["year"] = y
        frames.append(dfy)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ------------------------
# Merge logics
# ------------------------
def merge_strict_round(sack: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """Jointure stricte sur (ville+surface, round, joueurs triés) — pas de date."""
    odds_cols_td = [c for c in td.columns if re.match(r'^[A-Za-z0-9]+[WL]$', str(c))]
    cols_td = ["key_city_round_players","Tournament","Location","Surface","Surface_c","Round","rnd_canon","Winner","Loser","Date","Date_dt"] + odds_cols_td
    m = sack.merge(
        td[cols_td].rename(columns={"rnd_canon":"rnd_canon_td"}),
        on="key_city_round_players",
        how="left",
        suffixes=("_sack","_td")
    )
    return m

def fallback_by_nearest_date(m: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """Pour les non appariés: jointure sur (ville+surface, joueurs) puis choix de la date TD la plus proche (±DATE_WINDOW_DAYS), en préférant le même round."""
    need_idx = m.index[m["Tournament"].isna()]
    if len(need_idx) == 0:
        return m

    odds_cols_td = [c for c in td.columns if re.match(r'^[A-Za-z0-9]+[WL]$', str(c))]
    td_sel_cols = ["key_city_players","Date_dt","Tournament","Location","Surface","Surface_c","Round","rnd_canon","Winner","Loser","Date"] + odds_cols_td
    td_sub = td[td_sel_cols].copy()

    groups = {k: g.copy() for k, g in td_sub.groupby("key_city_players", sort=False)}
    filled = 0

    assign_cols = [c for c in td_sel_cols if c != "key_city_players"]
    for idx in need_idx:
        row = m.loc[idx]
        k = row["key_city_players"]
        d = row["date_tourney"]
        if k not in groups or pd.isna(d):
            continue
        cand = groups[k]
        cand = cand.assign(delta=(cand["Date_dt"] - d).abs())
        cand = cand[cand["delta"] <= pd.Timedelta(days=DATE_WINDOW_DAYS)]
        if cand.empty:
            continue
        rnd = row.get("rnd_canon", "")
        same_rnd = cand[cand["rnd_canon"] == rnd]
        if not same_rnd.empty:
            cand = same_rnd
        best = cand.sort_values("delta").iloc[0]
        for ccol in assign_cols:
            m.loc[idx, ccol] = best[ccol]
        filled += 1

    print(f"Remplis par fallback 'ville+joueurs + date proche' : {filled}")
    return m

def fallback_city_only_nearest_date(m: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """Dernier recours: jointure sur (VILLE seule, joueurs triés), date la plus proche (±DATE_WINDOW_CITY_ONLY)."""
    need_idx = m.index[m["Tournament"].isna()]
    if len(need_idx) == 0:
        return m

    odds_cols_td = [c for c in td.columns if re.match(r'^[A-Za-z0-9]+[WL]$', str(c))]
    td_sel_cols = ["key_cityonly_players","Date_dt","Tournament","Location","Surface","Surface_c","Round","rnd_canon","Winner","Loser","Date"] + odds_cols_td
    td_sub = td[td_sel_cols].copy()

    groups = {k: g.copy() for k, g in td_sub.groupby("key_cityonly_players", sort=False)}
    filled = 0

    assign_cols = [c for c in td_sel_cols if c != "key_cityonly_players"]
    for idx in need_idx:
        row = m.loc[idx]
        k = row.get("key_cityonly_players", "")
        d = row["date_tourney"]
        if not isinstance(k, str) or k == "" or pd.isna(d) or k not in groups:
            continue
        cand = groups[k]
        cand = cand.assign(delta=(cand["Date_dt"] - d).abs())
        cand = cand[cand["delta"] <= pd.Timedelta(days=DATE_WINDOW_CITY_ONLY)]
        if cand.empty:
            continue
        rnd = row.get("rnd_canon", "")
        same_rnd = cand[cand["rnd_canon"] == rnd]
        if not same_rnd.empty:
            cand = same_rnd
        best = cand.sort_values("delta").iloc[0]
        for ccol in assign_cols:
            m.loc[idx, ccol] = best[ccol]
        filled += 1

    print(f"Remplis par fallback 'VILLE seule + joueurs + date proche' : {filled}")
    return m

# ------------------------
# MAIN
# ------------------------
def main():
    sack = load_sackmann(YEARS)
    td = load_tennisdata(YEARS)

    print(f"Sackmann rows: {len(sack)} | Tennis-data rows: {len(td)}")

    # Filtrer sur rounds principaux (évite qualifs/exhibitions)
    sack = sack[sack["rnd_canon"].isin(MAIN_ROUNDS)].copy()
    td = td[td["rnd_canon"].isin(MAIN_ROUNDS)].copy()

    # 1) Jointure stricte (ville+surface + round + joueurs)
    m = merge_strict_round(sack, td)
    print("Après jointure stricte :", m["Tournament"].notna().sum(), "/", len(m))

    # 2) Fallback par proximité de date (ville+surface + joueurs)
    m = fallback_by_nearest_date(m, td)
    print("Après fallback proximité date :", m["Tournament"].notna().sum(), "/", len(m))

    # 3) Fallback VILLE seule (sans surface) ±14 jours
    m = fallback_city_only_nearest_date(m, td)
    print("Après fallback VILLE seule :", m["Tournament"].notna().sum(), "/", len(m))

    # 4) Détection propre des colonnes d'odds (regex)
    odds_regex = re.compile(r'^[A-Za-z0-9]+[WL]$')
    odds_cols = [c for c in td.columns if odds_regex.match(str(c))]
    odds_cols = [c for c in odds_cols if c in m.columns]

    # Rankings historiques : à intégrer plus tard via merge_asof
    rank_files = sorted(SACK.glob("atp_rankings_*.csv"))
    if rank_files:
        print("Note: rankings historiques détectés mais non intégrés dans cette version (on fera un merge_asof dédié).")
    else:
        print("Rankings historiques non trouvés (atp_rankings_*.csv) — étape ignorée (OK).")

    # Sélection & export
    base_cols = [
        "date_tourney","tourney_name","surface","round","winner_name","loser_name","score",
        "Tournament","Location","Surface","Round","Winner","Loser","Date"
    ]
    tech_cols = [
        "key_city_players","key_city_round_players","key_cityonly_players","key_cityonly_round_players",
        "tkey_city","city_key","players_sorted","rnd_canon"
    ]
    # Ajoute un paquet de colonnes Sackmann utiles si elles existent
    sack_extra = [c for c in m.columns if (c.startswith(("w_","l_","winner_","loser_")) or c in {"best_of","minutes"})]

    keep = [c for c in base_cols if c in m.columns] \
         + odds_cols \
         + [c for c in tech_cols if c in m.columns] \
         + [c for c in sack_extra if c in m.columns]

    final = m[keep].copy()

    final = final.rename(columns={
        "date_tourney":"tourney_start_date",
        "tourney_name":"tourney_sack",
        "surface":"surface_sack",
        "round":"round_sack",
        "Tournament":"tourney_td",
        "Surface":"surface_td",
        "Round":"round_td",
        "Winner":"winner_td",
        "Loser":"loser_td",
        "Date":"match_date_td"
    })

    out_path = OUT / "matches_final2.csv"
    final.to_csv(out_path, index=False)
    print(f"OK - fichier écrit : {out_path}")


    # Diagnostics odds (réels)
    if odds_cols:
        with_odds = final[odds_cols].notna().any(axis=1).sum()
        missing_odds = final[odds_cols].isna().all(axis=1).sum()
        print("Appariés avec ≥1 cote:", with_odds, "/", len(final))
        print("Sans aucune cote:", missing_odds)
        if missing_odds:
            final[final[odds_cols].isna().all(axis=1)].head(30).to_csv(CACHE / "sample_missing_odds.csv", index=False)
            print("Exemples sans cotes -> work/cache/sample_missing_odds.csv")
    else:
        print("Aucune colonne d'odds détectée (vérifie le schéma Tennis-Data).")

    # Echantillon non apparié pour debug
    not_matched = final[final["tourney_td"].isna()]
    if len(not_matched):
        not_matched_cols = [c for c in ["tourney_sack","surface_sack","round_sack","winner_name","loser_name"] if c in final.columns]
        final.loc[not_matched.index, not_matched_cols].head(30).to_csv(CACHE / "unmatched_sample.csv", index=False)
        print("Échantillon non apparié -> work/cache/unmatched_sample.csv")

if __name__ == "__main__":
    main()
