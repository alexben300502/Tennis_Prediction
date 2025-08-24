# make_features.py
# ------------------------------------------------------------
# Construit un dataset de features pré-match orienté "favori" (A vs B)
# Entrée : work/outputs/matches_final2.csv (issu de merge_tennis.py)
# Sortie : work/outputs/features_v1.csv
# ------------------------------------------------------------

from pathlib import Path
import re
import numpy as np
import pandas as pd

IN_PATH  = Path("work/outputs/matches_final2.csv")
OUT_PATH = Path("work/outputs/features_v1.csv")

# --------- Petites constantes ----------
ROUND_ORD = {"R128":1,"R64":2,"R32":3,"R16":4,"QF":5,"SF":6,"F":7}
BOOK_PRIORITY = ["PS", "Avg", "B365", "Max", "LB", "EX"]  # ordre de préférence
SURFACES = {"hard","clay","grass","carpet"}

# --------- Aides odds/probas ----------
def pick_book_pair(row):
    """Choisit (book, cotes W/L) selon ordre de priorité, si les deux cotes existent et >1."""
    for b in BOOK_PRIORITY:
        w, l = f"{b}W", f"{b}L"
        if w in row and l in row:
            ow, ol = row[w], row[l]
            if pd.notna(ow) and pd.notna(ol) and ow > 1 and ol > 1:
                return b, float(ow), float(ol)
    return None, np.nan, np.nan

def implied_probs_2way(odds_w, odds_l):
    """Calcule les probabilités implicites normalisées (2 issues), corrige l'overround."""
    pw_raw = 1.0/odds_w
    pl_raw = 1.0/odds_l
    s = pw_raw + pl_raw
    if s <= 0:
        return np.nan, np.nan
    pw = pw_raw / s
    pl = 1.0 - pw
    return pw, pl

# --------- Choix surface / date ----------
def choose_surface(row):
    s = row.get("surface_td")
    if isinstance(s, str) and s:
        return s
    s = row.get("surface_sack")
    return s if isinstance(s, str) else ""

def choose_match_date(row):
    # date tennis-data = date réelle du match; fallback = début de tournoi Sackmann
    d = row.get("match_date_td")
    if pd.notna(d):
        return d
    return row.get("tourney_start_date")

# --------- Elo ----------
def compute_elo_pre(df_matches, k=32, init=1500.0):
    """
    Calcule Elo global + Elo par surface AVANT le match (pas de fuite).
    df_matches: colonnes ['match_id','match_date','w_key','l_key','surface_final'] triées temporellement.
    """
    elo_global = {}
    elo_surface = {"hard":{}, "clay":{}, "grass":{}, "carpet":{}}

    pre_w, pre_l, pre_ws, pre_ls = [], [], [], []
    dm = df_matches.sort_values(["match_date","match_id"]).reset_index(drop=True)

    for _, r in dm.iterrows():
        w, l = r["w_key"], r["l_key"]
        surf = r["surface_final"] if r["surface_final"] in SURFACES else "hard"

        ew  = elo_global.get(w, init)
        el  = elo_global.get(l, init)
        ews = elo_surface[surf].get(w, init)
        els = elo_surface[surf].get(l, init)

        # stocker les Elo "pré-match"
        pre_w.append(ew); pre_l.append(el); pre_ws.append(ews); pre_ls.append(els)

        # attentes
        pw  = 1.0 / (1.0 + 10 ** ((el - ew)/400.0))
        pws = 1.0 / (1.0 + 10 ** ((els - ews)/400.0))

        # updates (w gagne, l perd)
        elo_global[w] = ew  + k * (1 - pw)
        elo_global[l] = el  + k * (0 - (1 - pw))
        elo_surface[surf][w] = ews + k * (1 - pws)
        elo_surface[surf][l] = els + k * (0 - (1 - pws))

    dm["pre_elo_w"]  = pre_w
    dm["pre_elo_l"]  = pre_l
    dm["pre_selo_w"] = pre_ws
    dm["pre_selo_l"] = pre_ls
    return dm[["match_id","pre_elo_w","pre_elo_l","pre_selo_w","pre_selo_l"]]

# --------- MAIN PIPELINE ----------
def main():
    # 0) Load
    df = pd.read_csv(IN_PATH, low_memory=False)
    for c in ["match_date_td","tourney_start_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 1) Date & surface "finales"
    df["match_date"]    = df.apply(choose_match_date, axis=1)
    df["surface_final"] = df.apply(choose_surface, axis=1).fillna("").astype(str)
    # normalise en minuscule pour les comparaisons et l'Elo par surface
    df["surface_final"] = df["surface_final"].str.lower()

    df = df[df["match_date"].notna()].copy()
    df = df.reset_index(drop=True)
    df["match_id"] = df.index  # id stable

    # 2) Sanity: clés joueurs (exportées par merge_tennis)
    if "w_key" not in df.columns or "l_key" not in df.columns:
        raise RuntimeError("Colonnes 'w_key' / 'l_key' absentes. Relance 'merge_tennis.py' avec l'export sack_extra.")

    # 3) Round -> ordinal (si rnd_canon dispo)
    if "rnd_canon" in df.columns:
        df["round_ord"] = df["rnd_canon"].map(ROUND_ORD).fillna(0).astype(int)
    else:
        df["round_ord"] = 0

    # 4) Choix du bookmaker & probas implicites (corrigées)
    books, oW, oL, pW, pL = [], [], [], [], []
    for _, r in df.iterrows():
        b, ow, ol = pick_book_pair(r)
        books.append(b); oW.append(ow); oL.append(ol)
        if b is None or not np.isfinite(ow) or not np.isfinite(ol):
            pW.append(np.nan); pL.append(np.nan)
        else:
            pw, pl = implied_probs_2way(ow, ol)
            pW.append(pw); pL.append(pl)

    df["book_used"] = books
    df["oddsW"], df["oddsL"] = oW, oL
    df["pW_book"], df["pL_book"] = pW, pL

    # 5) Orientation A/B (A = favori) + cible y (1 si A gagne)
    A_key, B_key, y = [], [], []
    for _, r in df.iterrows():
        if r["book_used"] is None or not np.isfinite(r["oddsW"]) or not np.isfinite(r["oddsL"]):
            A_key.append(np.nan); B_key.append(np.nan); y.append(np.nan)
            continue
        fav_is_winner = r["oddsW"] <= r["oddsL"]
        if fav_is_winner:
            A_key.append(r["w_key"]); B_key.append(r["l_key"]); y.append(1)
        else:
            A_key.append(r["l_key"]); B_key.append(r["w_key"]); y.append(0)

    df["A_key"] = A_key
    df["B_key"] = B_key
    df["y"]     = y

    # Garder uniquement les lignes avec un book valide
    df = df[df["book_used"].notna()].copy()
    # Proba du favori A (normalisée)
    df["p_book_A"] = np.where(df["oddsW"] <= df["oddsL"], df["pW_book"], df["pL_book"])
    df["odds_A"]   = np.where(df["oddsW"] <= df["oddsL"], df["oddsW"], df["oddsL"])
    df["odds_B"]   = np.where(df["oddsW"] <= df["oddsL"], df["oddsL"], df["oddsW"])

    # 6) Tableau "long" joueur-match pour calculs séquentiels
    #    (forme rolling & repos) — on n'utilise que l'historique AVANT le match.
    long_rows = []
    for _, r in df.iterrows():
        # gagnant
        long_rows.append({"match_id": r["match_id"], "date": r["match_date"], "surface": r["surface_final"],
                          "player_key": r["w_key"], "opp_key": r["l_key"], "is_win": 1})
        # perdant
        long_rows.append({"match_id": r["match_id"], "date": r["match_date"], "surface": r["surface_final"],
                          "player_key": r["l_key"], "opp_key": r["w_key"], "is_win": 0})
    long = pd.DataFrame(long_rows).dropna(subset=["player_key"]).copy()
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.sort_values(["player_key","date","match_id"]).reset_index(drop=True)

    # Forme rolling 5/10 (sans fuite : on shift avant rolling)
    long["form5"]  = long.groupby("player_key", sort=False)["is_win"] \
                        .transform(lambda s: s.shift().rolling(5,  min_periods=1).mean())

    long["form10"] = long.groupby("player_key", sort=False)["is_win"] \
                        .transform(lambda s: s.shift().rolling(10, min_periods=1).mean())

    # Repos (jours depuis dernier match)
    long["rest_days"] = long.groupby("player_key")["date"].diff().dt.days
    long["rest_days"] = long["rest_days"].fillna(14).clip(lower=0, upper=60)

    # 7) H2H avant match
    ps = np.where(long["player_key"] <= long["opp_key"], long["player_key"], long["opp_key"])
    pb = np.where(long["player_key"]  > long["opp_key"], long["player_key"], long["opp_key"])
    long = long.assign(pair_small=ps, pair_big=pb, is_small=(long["player_key"] == ps))
    long = long.sort_values(["pair_small","pair_big","date","match_id"])

    long["win_small_this"] = ((long["is_win"] == 1) & (long["is_small"])).astype(int)
    long["win_big_this"]   = ((long["is_win"] == 1) & (~long["is_small"])).astype(int)

    long["h2h_small_before"] = long.groupby(["pair_small","pair_big"])["win_small_this"].cumsum().shift().fillna(0).astype(int)
    long["h2h_big_before"]   = long.groupby(["pair_small","pair_big"])["win_big_this"].cumsum().shift().fillna(0).astype(int)

    long["h2h_player_before"] = np.where(long["is_small"], long["h2h_small_before"], long["h2h_big_before"])
    long["h2h_opp_before"]    = np.where(long["is_small"], long["h2h_big_before"], long["h2h_small_before"])
    long["h2h_total_before"]  = long["h2h_player_before"] + long["h2h_opp_before"]
    long["h2h_wr_before"]     = np.where(long["h2h_total_before"] > 0,
                                         long["h2h_player_before"] / long["h2h_total_before"], 0.5)

    # 8) Elo global + surface (valeur "pré-match")
    elo_input = df[["match_id","match_date","w_key","l_key","surface_final"]].dropna(subset=["w_key","l_key"]).copy()
    elo_vals  = compute_elo_pre(elo_input, k=32, init=1500.0)
    df = df.merge(elo_vals, on="match_id", how="left")

    # 9) Jointure des features joueur -> A/B
    pcols = ["match_id","player_key","form5","form10","rest_days",
             "h2h_player_before","h2h_opp_before","h2h_total_before","h2h_wr_before"]
    pf = long[pcols].copy()

    df = df.merge(
        pf.rename(columns={"player_key":"A_key",
                           "form5":"A_form5","form10":"A_form10","rest_days":"A_rest",
                           "h2h_player_before":"A_h2h_wins_before",
                           "h2h_opp_before":"A_h2h_opp_before",
                           "h2h_total_before":"A_h2h_total_before",
                           "h2h_wr_before":"A_h2h_wr_before"}),
        on=["match_id","A_key"], how="left"
    )
    df = df.merge(
        pf.rename(columns={"player_key":"B_key",
                           "form5":"B_form5","form10":"B_form10","rest_days":"B_rest",
                           "h2h_player_before":"B_h2h_wins_before",
                           "h2h_opp_before":"B_h2h_opp_before",
                           "h2h_total_before":"B_h2h_total_before",
                           "h2h_wr_before":"B_h2h_wr_before"}),
        on=["match_id","B_key"], how="left"
    )

    # 10) Rangs/âges A/B (depuis winner_*/loser_* si dispo), sans fuite
    def pick_side_vals(row, w_col, l_col):
        wv, lv = row.get(w_col), row.get(l_col)
        if row["oddsW"] <= row["oddsL"]:  # favori = winner
            return wv, lv
        else:
            return lv, wv

    for base in ["rank","rank_points","age","ht","hand"]:
        wcol, lcol = f"winner_{base}", f"loser_{base}"
        if wcol in df.columns and lcol in df.columns:
            A_vals, B_vals = zip(*df.apply(lambda r: pick_side_vals(r, wcol, lcol), axis=1))
            df[f"A_{base}"] = A_vals
            df[f"B_{base}"] = B_vals

    # 11) Elo A/B + diffs
    df["A_elo"]  = np.where(df["oddsW"] <= df["oddsL"], df["pre_elo_w"],  df["pre_elo_l"])
    df["B_elo"]  = np.where(df["oddsW"] <= df["oddsL"], df["pre_elo_l"],  df["pre_elo_w"])
    df["A_selo"] = np.where(df["oddsW"] <= df["oddsL"], df["pre_selo_w"], df["pre_selo_l"])
    df["B_selo"] = np.where(df["oddsW"] <= df["oddsL"], df["pre_selo_l"], df["pre_selo_w"])

    df["elo_diff"]  = df["A_elo"]  - df["B_elo"]
    df["selo_diff"] = df["A_selo"] - df["B_selo"]

    if "A_rank" in df.columns and "B_rank" in df.columns:
        # signe positif si A mieux classé (rang plus petit)
        df["rank_diff"] = (df["B_rank"].fillna(5000) - df["A_rank"].fillna(5000))
    if "A_age" in df.columns and "B_age" in df.columns:
        df["age_diff"]  = df["A_age"].fillna(0) - df["B_age"].fillna(0)

    df["rest_diff"]   = df["A_rest"].fillna(14) - df["B_rest"].fillna(14)
    df["form5_diff"]  = df["A_form5"].fillna(0.5) - df["B_form5"].fillna(0.5)
    df["form10_diff"] = df["A_form10"].fillna(0.5) - df["B_form10"].fillna(0.5)

    # 12) Encodage surface + round
    df["round_ord"] = df["round_ord"].fillna(0).astype(int)
    surf = df["surface_final"].fillna("").astype(str)
    for val in ["hard","clay","grass","carpet"]:
        df[f"surf_{val}"] = (surf == val).astype(int)

    # 13) best_of si présent
    if "best_of" in df.columns:
        df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce")

    # 14) Colonnes finales : métadonnées + features
    meta_cols = ["match_id","match_date","tourney_sack","tourney_td","round_sack","round_td",
                 "surface_final","book_used","A_key","B_key","y","p_book_A","odds_A","odds_B"]
    feat_cols = [
        "A_elo","B_elo","elo_diff","A_selo","B_selo","selo_diff",
        "A_form5","B_form5","form5_diff","A_form10","B_form10","form10_diff",
        "A_rest","B_rest","rest_diff","round_ord","surf_hard","surf_clay","surf_grass","surf_carpet"
    ]
    # optionnels si dispo
    for opt in ["A_rank","B_rank","rank_diff","A_age","B_age","age_diff","best_of",
                "A_h2h_wins_before","B_h2h_wins_before","A_h2h_total_before","B_h2h_total_before",
                "A_h2h_wr_before","B_h2h_wr_before"]:
        if opt in df.columns:
            feat_cols.append(opt)

    keep = [c for c in meta_cols if c in df.columns] + feat_cols
    out = df[keep].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("OK - features écrites :", OUT_PATH)
    print("Lignes :", len(out))
    print("Cible non nulle (y):", out["y"].notna().sum())
    print("Couverture book (p_book_A non nulle):", out["p_book_A"].notna().sum())

if __name__ == "__main__":
    import pandas as pd
    main()
