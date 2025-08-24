# sanity_checks.py
# ------------------------------------------------------------
# Sanity checks pour détecter fuites & artefacts :
#  - Baseline marché vs modèle (identique à train_lgbm.py)
#  - PS-only : évaluer uniquement les matchs Pinnacle (book_used == "PS")
#  - Permutation test : réentraîner avec y permuté -> logloss ~ 0.693 attendu
#  - Ablation de features : retirer Elo/H2H/Forme/Rest (ne garder que features "basiques")
#  - Top single-feature AUC : repérer une feature “magique” (potentielle fuite)
# Sorties : métriques + tables dans work/outputs/sanity_*.*.
# ------------------------------------------------------------

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import joblib

IN  = Path("work/outputs/features_v1.csv")
OUT = Path("work/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
def metric_block(name, y, p):
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    m = {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "n": int(len(y))
    }
    print(f"{name:<28} | LogLoss={m['logloss']:.4f} | Brier={m['brier']:.4f} | AUC={m['auc']:.4f} | n={m['n']}")
    return m

def calib_table(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    dfp = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
    dfp["bin"] = pd.qcut(dfp.index, q=n_bins, labels=False, duplicates="drop")
    out = dfp.groupby("bin").agg(
        n=("y","size"), p_mean=("p","mean"), y_rate=("y","mean"),
        p_min=("p","min"), p_max=("p","max")
    ).reset_index()
    out["gap"] = out["p_mean"] - out["y_rate"]
    return out

def build_feature_matrix(df: pd.DataFrame, mode: str):
    """
    mode:
      - 'full'  = toutes les features numériques pré-match (comme train_lgbm.py)
      - 'basic' = ablation (retire Elo/H2H/Forme/Repos), ne garde que round/surface/rank/age (+ diffs)/best_of
    """
    exclude = {
        "match_id","match_date","tourney_sack","tourney_td","round_sack","round_td",
        "surface_final","A_key","B_key","book_used","odds_A","odds_B","p_book_A","y"
    }
    if mode == "full":
        num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    elif mode == "basic":
        keep_prefix = ("round_ord","surf_","A_rank","B_rank","rank_diff","A_age","B_age","age_diff")
        num_cols = [c for c in df.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])
                    and (c.startswith(keep_prefix) or c in ("best_of",))]
    else:
        raise ValueError("mode must be 'full' or 'basic'")
    X = df[num_cols].copy()
    return num_cols, X

def fillna_with_train_stats(Xtr, *X_others):
    med = Xtr.median(numeric_only=True)
    Xtr_f = Xtr.fillna(med)
    outs = [Xtr_f]
    for X in X_others:
        missing = [c for c in med.index if c not in X.columns]
        for c in missing:
            X[c] = np.nan
        X = X[med.index]
        outs.append(X.fillna(med))
    return tuple(outs)

def train_and_eval(df, mode="full", ps_only=False, seed=42):
    # splits temporels
    train = df[df["match_date"].dt.year <= 2022].copy()
    val   = df[df["match_date"].dt.year == 2023].copy()
    test  = df[df["match_date"].dt.year == 2024].copy()
    assert len(train) and len(val) and len(test), "Splits vides."

    # y / book
    ytr = train["y"].to_numpy(dtype=np.int32)
    yv  = val["y"].to_numpy(dtype=np.int32)
    yte = test["y"].to_numpy(dtype=np.int32)

    p_book_val = val["p_book_A"].to_numpy(dtype=np.float64)
    p_book_tst = test["p_book_A"].to_numpy(dtype=np.float64)

    # PS-only mask (pour l'éval uniquement)
    mask_val_ps  = (val["book_used"]  == "PS").to_numpy()
    mask_test_ps = (test["book_used"] == "PS").to_numpy()

    # Features
    feat_cols, Xtr = build_feature_matrix(train, mode)
    _, Xv  = build_feature_matrix(val, mode)
    _, Xte = build_feature_matrix(test, mode)
    Xtr_f, Xv_f, Xte_f = fillna_with_train_stats(Xtr, Xv, Xte)

    # Baseline marché
    print("\n== Baseline marché ==")
    metrics_book_val = metric_block("Val 2023 (book)", yv,  p_book_val)
    metrics_book_tst = metric_block("Test 2024 (book)", yte, p_book_tst)

    if ps_only:
        print("\n== Baseline marché (PS only) ==")
        metric_block("Val 2023 PS (book)",  yv[mask_val_ps],  p_book_val[mask_val_ps])
        metric_block("Test 2024 PS (book)", yte[mask_test_ps], p_book_tst[mask_test_ps])

    # LightGBM
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        random_state=seed
    )
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        Xtr_f, ytr,
        eval_set=[(Xv_f, yv)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)]
    )

    # Calibration isotonic
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(Xv_f, yv)

    p_val  = cal.predict_proba(Xv_f)[:,1]
    p_test = cal.predict_proba(Xte_f)[:,1]

    print("\n== Modèle calibré ==")
    metrics_model_val = metric_block("Val 2023 (model)", yv,  p_val)
    metrics_model_tst = metric_block("Test 2024 (model)", yte, p_test)

    if ps_only:
        print("\n== Modèle calibré (PS only) ==")
        metric_block("Val 2023 PS (model)",  yv[mask_val_ps],  p_val[mask_val_ps])
        metric_block("Test 2024 PS (model)", yte[mask_test_ps], p_test[mask_test_ps])

    # Importances
    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": clf.booster_.feature_importance(importance_type="gain"),
        "split": clf.booster_.feature_importance(importance_type="split")
    }).sort_values("gain", ascending=False)
    suffix = f"_{mode}" + ("_PS" if ps_only else "")
    imp.to_csv(OUT/f"sanity_feature_importances{suffix}.csv", index=False)

    # Calibration tables (modèle)
    calib_val_model  = calib_table(yv,  p_val,  10)
    calib_test_model = calib_table(yte, p_test, 10)
    calib_val_model.to_csv(OUT/f"sanity_calibration_val_model{suffix}.csv", index=False)
    calib_test_model.to_csv(OUT/f"sanity_calibration_test_model{suffix}.csv", index=False)

    # Deltas vs marché
    d_val_ll = metrics_book_val["logloss"] - metrics_model_val["logloss"]
    d_val_br = metrics_book_val["brier"]   - metrics_model_val["brier"]
    d_tst_ll = metrics_book_tst["logloss"] - metrics_model_tst["logloss"]
    d_tst_br = metrics_book_tst["brier"]   - metrics_model_tst["brier"]
    print(f"\nΔLogLoss val  (book - model) = {d_val_ll:+.4f} | ΔBrier val = {d_val_br:+.4f}")
    print(f"ΔLogLoss test (book - model) = {d_tst_ll:+.4f} | ΔBrier test = {d_tst_br:+.4f}")

    # Sauvegarde du modèle calibré (pour ce run)
    joblib.dump(cal, OUT/f"sanity_model_calibrated{suffix}.pkl")

    # -------- Permutation test (Option A : early stopping avec eval_set + eval_metric) --------
    rng = np.random.default_rng(42)
    ytr_perm = rng.permutation(ytr)

    clf_p = lgb.LGBMClassifier(**params)
    clf_p.fit(
        Xtr_f, ytr_perm,
        eval_set=[(Xv_f, yv)],      # jeu de validation fourni
        eval_metric="logloss",      # métrique fournie
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    p_perm = clf_p.predict_proba(Xv_f)[:,1]
    print("\nPermutation check (devrait ≈ 0.693) :", log_loss(yv, p_perm))

    # Top single-feature AUC (diag fuite) sur validation
    aucs = []
    for c in Xv_f.columns:
        try:
            aucs.append((c, abs(roc_auc_score(yv, Xv_f[c]) - 0.5)))
        except Exception:
            pass
    top = sorted(aucs, key=lambda x: x[1], reverse=True)[:25]
    pd.DataFrame(top, columns=["feature","|AUC-0.5|"]).to_csv(OUT/f"sanity_top_single_feature_auc{suffix}.csv", index=False)

    return {
        "metrics_book_val": metrics_book_val,
        "metrics_book_tst": metrics_book_tst,
        "metrics_model_val": metrics_model_val,
        "metrics_model_tst": metrics_model_tst,
        "delta_val_logloss": float(d_val_ll),
        "delta_test_logloss": float(d_tst_ll),
        "delta_val_brier": float(d_val_br),
        "delta_test_brier": float(d_tst_br)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["full","basic"], default="full",
                        help="full = toutes features ; basic = sans Elo/H2H/Forme/Rest")
    parser.add_argument("--ps-only", action="store_true",
                        help="évaluer uniquement les matchs Pinnacle (PS)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(IN, parse_dates=["match_date"])

    # Garde-fous : signaler si des colonnes post-match sont présentes (ne devraient PAS être utilisées ici)
    forbidden = [c for c in df.columns if c in {"minutes","score"} or c.startswith(("w_","l_"))]
    if forbidden:
        print("⚠️ ATTENTION : colonnes potentiellement post-match détectées dans features_v1 (elles ne seront pas utilisées) :", forbidden)

    out = train_and_eval(df, mode=args.ablation, ps_only=args.ps_only, seed=args.seed)
    pd.Series(out).to_json(OUT/f"sanity_metrics_{args.ablation}{'_PS' if args.ps_only else ''}.json")
    print("\nOK - sanity checks terminés.")

if __name__ == "__main__":
    main()
