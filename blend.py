# blend.py
# ------------------------------------------------------------
# Blending p_model avec p_book_A (marché) :
# - récupère p_book_A depuis pred_* ou features_v1.csv
# - si absent, recalcule depuis odds_A/B
# - optimise w sur VAL (2023), évalue sur TEST (2024)
# Sorties : pred_val_blend.csv, pred_test_blend.csv, blend_summary.json
# ------------------------------------------------------------
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

OUT = Path("work/outputs")
PRED_VAL  = OUT / "pred_val.csv"
PRED_TEST = OUT / "pred_test.csv"
FEAT_CSV  = OUT / "features_v2.csv"

def metrics(df, pcol):
    y = df["y"].astype(int).values
    p = df[pcol].astype(float).clip(1e-6, 1-1e-6).values
    return {
        "LogLoss": float(log_loss(y, p)),
        "Brier":   float(brier_score_loss(y, p)),
        "AUC":     float(roc_auc_score(y, p)),
        "n":       int(len(df)),
    }

def attach_book_probs(df_pred: pd.DataFrame, df_feat: pd.DataFrame | None) -> pd.DataFrame:
    """S'assure que df_pred contient p_book_A; sinon, l'ajoute depuis features ou calcule via odds."""
    df = df_pred.copy()
    # 1) déjà présent ?
    for cand in ["p_book_A", "p_book", "p_market"]:
        if cand in df.columns:
            if cand != "p_book_A":
                df = df.rename(columns={cand: "p_book_A"})
            return df

    # 2) essayer depuis features_v1.csv (même ordre / même longueur)
    if df_feat is None and FEAT_CSV.exists():
        df_feat = pd.read_csv(FEAT_CSV, low_memory=False)

    if df_feat is not None and len(df_feat) == len(df):
        if "p_book_A" in df_feat.columns:
            df["p_book_A"] = df_feat["p_book_A"].astype(float).values
            return df
        # 3) sinon, tenter via odds
        if {"odds_A","odds_B"} <= set(df_feat.columns):
            oa = pd.to_numeric(df_feat["odds_A"], errors="coerce").values
            ob = pd.to_numeric(df_feat["odds_B"], errors="coerce").values
            pA = (1.0/oa) / ((1.0/oa) + (1.0/ob))
            df["p_book_A"] = pA
            return df

    # 4) ultime chance : les odds sont dans df_pred (peu probable)
    if {"odds_A","odds_B"} <= set(df.columns):
        oa = pd.to_numeric(df["odds_A"], errors="coerce").values
        ob = pd.to_numeric(df["odds_B"], errors="coerce").values
        pA = (1.0/oa) / ((1.0/oa) + (1.0/ob))
        df["p_book_A"] = pA
        return df

    raise KeyError("Impossible de trouver ou reconstruire p_book_A. "
                   "Vérifie que work/outputs/features_v1.csv existe et correspond bien aux prédictions.")

def ensure_model_prob(df: pd.DataFrame) -> pd.DataFrame:
    """S'assure que df contient la colonne des proba modèle 'p_model'."""
    df = df.copy()
    if "p_model" in df.columns:
        return df
    # quelques alias potentiels
    for cand in ["p_hat", "p_pred", "prob_model", "p"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "p_model"})
            return df
    raise KeyError("Colonne 'p_model' introuvable dans les CSV de prédictions.")

def search_best_weight(df_val: pd.DataFrame) -> float:
    best = None
    for w in np.linspace(0, 1, 41):  # pas 0.025
        p = (w*df_val["p_model"].values + (1-w)*df_val["p_book_A"].values).clip(1e-6, 1-1e-6)
        m = {
            "LogLoss": log_loss(df_val["y"].astype(int).values, p),
        }
        if best is None or m["LogLoss"] < best[0]:
            best = (m["LogLoss"], w)
    return best[1] if best else 0.5

def main():
    if not PRED_VAL.exists() or not PRED_TEST.exists():
        raise FileNotFoundError("pred_val.csv ou pred_test.csv manquant dans work/outputs/. Lance d'abord train_lgbm.py")

    pred_val  = pd.read_csv(PRED_VAL,  low_memory=False)
    pred_test = pd.read_csv(PRED_TEST, low_memory=False)
    feats = pd.read_csv(FEAT_CSV, low_memory=False) if FEAT_CSV.exists() else None

    # Harmonise colonnes indispensables
    for df in [pred_val, pred_test]:
        if "y" not in df.columns:
            # parfois on a 'target' ou 'label'
            for cand in ["target", "label"]:
                if cand in df.columns:
                    df.rename(columns={cand:"y"}, inplace=True)
                    break
        if "y" not in df.columns:
            raise KeyError("Colonne 'y' introuvable dans pred_*.csv")

    pred_val  = ensure_model_prob(pred_val)
    pred_test = ensure_model_prob(pred_test)

    pred_val  = attach_book_probs(pred_val,  feats)
    pred_test = attach_book_probs(pred_test, feats)

    # Baselines
    mb_val  = metrics(pred_val,  "p_book_A")
    mb_test = metrics(pred_test, "p_book_A")
    mm_val  = metrics(pred_val,  "p_model")
    mm_test = metrics(pred_test, "p_model")

    print(f"Book VAL  | LogLoss={mb_val['LogLoss']:.4f} | Brier={mb_val['Brier']:.4f} | AUC={mb_val['AUC']:.4f} | n={mb_val['n']}")
    print(f"Book TEST | LogLoss={mb_test['LogLoss']:.4f} | Brier={mb_test['Brier']:.4f} | AUC={mb_test['AUC']:.4f} | n={mb_test['n']}")
    print(f"Model VAL | LogLoss={mm_val['LogLoss']:.4f} | Brier={mm_val['Brier']:.4f} | AUC={mm_val['AUC']:.4f}")
    print(f"Model TEST| LogLoss={mm_test['LogLoss']:.4f} | Brier={mm_test['Brier']:.4f} | AUC={mm_test['AUC']:.4f}")

    # Blend
    w = search_best_weight(pred_val)
    pred_val["p_blend"]  = np.clip(w*pred_val["p_model"]  + (1-w)*pred_val["p_book_A"], 1e-6, 1-1e-6)
    pred_test["p_blend"] = np.clip(w*pred_test["p_model"] + (1-w)*pred_test["p_book_A"], 1e-6, 1-1e-6)

    mbld_val  = metrics(pred_val.rename(columns={"p_blend":"p"}),  "p")
    mbld_test = metrics(pred_test.rename(columns={"p_blend":"p"}), "p")

    print(f"\n== Blend w={w:.3f} (optimisé sur VAL) ==")
    print(f"Blend VAL  | LogLoss={mbld_val['LogLoss']:.4f} | Brier={mbld_val['Brier']:.4f} | AUC={mbld_val['AUC']:.4f}")
    print(f"Blend TEST | LogLoss={mbld_test['LogLoss']:.4f} | Brier={mbld_test['Brier']:.4f} | AUC={mbld_test['AUC']:.4f}")

    # Sauvegardes
    pred_val.to_csv(OUT/"pred_val_blend.csv", index=False)
    pred_test.to_csv(OUT/"pred_test_blend.csv", index=False)
    with open(OUT/"blend_summary.json","w") as f:
        json.dump({
            "w": w,
            "book_val": mb_val, "book_test": mb_test,
            "model_val": mm_val, "model_test": mm_test,
            "blend_val": mbld_val, "blend_test": mbld_test
        }, f, indent=2)
    print("\nOK - saved pred_val_blend.csv, pred_test_blend.csv, blend_summary.json")

if __name__ == "__main__":
    main()
