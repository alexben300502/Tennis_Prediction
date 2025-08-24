# train_lgbm.py
# ------------------------------------------------------------
# Baseline modèle vs marché : LightGBM + calibration isotonic
# Splits : Train 2017–2022, Val 2023, Test 2024
# Sorties : métriques, prédictions, importances, calibration, modèle pickle
# ------------------------------------------------------------

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import joblib

IN  = Path("work/outputs/features_v2.csv")
OUT = Path("work/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
def metric_block(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    """Affiche et renvoie LogLoss/Brier/AUC."""
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    m = {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "n": int(len(y))
    }
    print(f"{name:<24} | LogLoss={m['logloss']:.4f} | Brier={m['brier']:.4f} | AUC={m['auc']:.4f} | n={m['n']}")
    return m

def calib_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Table de calibration (bins ~ égaux en taille)."""
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    dfp = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
    dfp["bin"] = pd.qcut(dfp.index, q=n_bins, labels=False, duplicates="drop")
    out = dfp.groupby("bin").agg(
        n=("y","size"),
        p_mean=("p","mean"),
        y_rate=("y","mean"),
        p_min=("p","min"),
        p_max=("p","max")
    ).reset_index()
    out["gap"] = out["p_mean"] - out["y_rate"]
    return out

def build_feature_matrix(df: pd.DataFrame):
    """Sélectionne les features numériques (pas de fuite)."""
    exclude = {
        "match_id","match_date","tourney_sack","tourney_td","round_sack","round_td",
        "surface_final","A_key","B_key","book_used","odds_A","odds_B","p_book_A","y"
    }
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].copy()
    return num_cols, X

def fillna_with_train_stats(Xtr: pd.DataFrame, *X_others: pd.DataFrame):
    """Impute NaN par médianes du train et aligne les colonnes sur les autres splits."""
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

# ----------------------------
# Load & splits temporels
# ----------------------------
df = pd.read_csv(IN, parse_dates=["match_date"])
if df.empty:
    raise RuntimeError("features_v1.csv est vide.")

train = df[df["match_date"].dt.year <= 2022].copy()
val   = df[df["match_date"].dt.year == 2023].copy()
test  = df[df["match_date"].dt.year == 2024].copy()
assert len(train) and len(val) and len(test), "Splits vides : vérifie les années dans features_v1.csv"

# y & baseline marché (np arrays explicites)
ytr = train["y"].to_numpy(dtype=np.int32)
yv  = val["y"].to_numpy(dtype=np.int32)
yte = test["y"].to_numpy(dtype=np.int32)

p_book_val = val["p_book_A"].to_numpy(dtype=np.float64)
p_book_tst = test["p_book_A"].to_numpy(dtype=np.float64)

# X
feat_cols, Xtr = build_feature_matrix(train)
_, Xv  = build_feature_matrix(val)
_, Xte = build_feature_matrix(test)

Xtr_f, Xv_f, Xte_f = fillna_with_train_stats(Xtr, Xv, Xte)

# ----------------------------
# Baseline marché
# ----------------------------
print("== Baseline marché (p_book_A) ==")
metrics_book_val = metric_block("Val 2023 (book)", yv,  p_book_val)
metrics_book_tst = metric_block("Test 2024 (book)", yte, p_book_tst)

calib_val_book  = calib_table(yv,  p_book_val, n_bins=10)
calib_test_book = calib_table(yte, p_book_tst, n_bins=10)
calib_val_book.to_csv(OUT/"calibration_val_book.csv", index=False)
calib_test_book.to_csv(OUT/"calibration_test_book.csv", index=False)

# ----------------------------
# LightGBM (callbacks au lieu de verbose=...)
# ----------------------------
params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=0.0,
    min_child_samples=20,
    random_state=42
)

clf = lgb.LGBMClassifier(**params)
clf.fit(
    Xtr_f, ytr,
    eval_set=[(Xv_f, yv)],
    eval_metric="logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=500, verbose=False),
        lgb.log_evaluation(period=0)  # silence logs
    ],
)

# Importances
imp = pd.DataFrame({
    "feature": feat_cols,
    "gain": clf.booster_.feature_importance(importance_type="gain"),
    "split": clf.booster_.feature_importance(importance_type="split")
}).sort_values("gain", ascending=False)
imp.to_csv(OUT/"feature_importances.csv", index=False)

# ----------------------------
# Calibration isotonic (sur validation 2023)
# ----------------------------
cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
cal.fit(Xv_f, yv)

# ----------------------------
# Évaluation finale
# ----------------------------
p_val  = cal.predict_proba(Xv_f)[:, 1]
p_test = cal.predict_proba(Xte_f)[:, 1]

print("\n== Modèle calibré ==")
metrics_model_val = metric_block("Val 2023 (model)", yv,  p_val)
metrics_model_tst = metric_block("Test 2024 (model)", yte, p_test)

def deltas(m_model, m_book, split_name):
    d_ll = m_book["logloss"] - m_model["logloss"]
    d_br = m_book["brier"]   - m_model["brier"]
    print(f"{split_name} : ΔLogLoss (book - model) = {d_ll:+.4f} | ΔBrier = {d_br:+.4f}")
    return {"d_logloss": float(d_ll), "d_brier": float(d_br)}

delta_val  = deltas(metrics_model_val, metrics_book_val, "Val 2023")
delta_test = deltas(metrics_model_tst, metrics_book_tst, "Test 2024")

# Tables de calibration modèle
calib_val_model  = calib_table(yv,  p_val,  n_bins=10)
calib_test_model = calib_table(yte, p_test, n_bins=10)
calib_val_model.to_csv(OUT/"calibration_val_model.csv", index=False)
calib_test_model.to_csv(OUT/"calibration_test_model.csv", index=False)

# ----------------------------
# (Option) Accuracy à seuil 0.5
# ----------------------------
def accuracy_at(p, y, thr=0.5):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    return float(((p >= thr).astype(int) == y).mean())

print("\n== Accuracy (seuil 0.5) ==")
print(f"Val 2023 (book)  acc = {accuracy_at(p_book_val, yv):.4f}")
print(f"Val 2023 (model) acc = {accuracy_at(p_val,      yv):.4f}")
print(f"Test 2024 (book) acc = {accuracy_at(p_book_tst, yte):.4f}")
print(f"Test 2024 (model) acc = {accuracy_at(p_test,     yte):.4f}")

# ----------------------------
# Sauvegardes
# ----------------------------
joblib.dump(cal, OUT/"model_lgbm_calibrated.pkl")

pd.DataFrame({"match_date": val["match_date"].values, "y": yv,  "p_book": p_book_val, "p_model": p_val}).to_csv(OUT/"pred_val.csv",  index=False)
pd.DataFrame({"match_date": test["match_date"].values,"y": yte, "p_book": p_book_tst,"p_model": p_test}).to_csv(OUT/"pred_test.csv", index=False)

pd.Series({
    "val_book": metrics_book_val,
    "test_book": metrics_book_tst,
    "val_model": metrics_model_val,
    "test_model": metrics_model_tst,
    "delta_val": delta_val,
    "delta_test": delta_test
}).to_json(OUT/"metrics_summary.json")

print("\nOK - saved:")
print("  ", OUT/"model_lgbm_calibrated.pkl")
print("  ", OUT/"feature_importances.csv")
print("  ", OUT/"pred_val.csv", "|", OUT/"pred_test.csv")
print("  ", OUT/"calibration_val_*", "|", OUT/"calibration_test_*")
print("  ", OUT/"metrics_summary.json")
