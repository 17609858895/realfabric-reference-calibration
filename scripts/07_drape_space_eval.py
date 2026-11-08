"""Step 07 - evaluate every estimator in drape space (the quantity designers see) and test drape-aware learning.

Scoring: drape coefficient (DC) of predicted parameters via a *cross-fitted* surrogate (the surrogate used for
fabric i never saw any scene of fabric i), against the DC of the real GT simulation.
Published EG2024 methods are scored both ways (real simulation of their parameters, and surrogate).

Learning variants (all metadata-only, trained on released labels):
  A  parameter-first   : 20 learners / stack -> 6 params -> surrogate DC
  B  drape-first       : metadata -> DC label (surrogate applied to the training labels)
  C  drape-weighted    : per-sample, per-target weights w_ij = (dDC/dlog k_j)^2 at the label point
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROC, RESULTS, TARGETS, duplicate_groups, engineer, load_raw, targets_log

warnings.filterwarnings("ignore")
SEED = 2024


def crossfit_surrogates(df_pairs: pd.DataFrame, feats: list[str], base) -> dict:
    """fabric ref -> surrogate trained on scenes of the other folds."""
    refs = df_pairs["ref"].to_numpy()
    out = {}
    for a, b in GroupKFold(5).split(df_pairs, groups=refs):
        m = clone(base).fit(df_pairs.iloc[a][feats].to_numpy(), df_pairs.iloc[a]["drape_coefficient"].to_numpy())
        for r in np.unique(refs[b]):
            out[r] = m
    return out


def dc_of(params_log: np.ndarray, log_density: np.ndarray, models: list) -> np.ndarray:
    X = np.column_stack([params_log, log_density])
    return np.array([m.predict(X[i:i + 1])[0] for i, m in enumerate(models)])


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr = targets_log(tr, "_MechMet")
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    feats, full = sur["features"], sur["model"]
    cf = crossfit_surrogates(pairs, feats, full)
    models_te = [cf[r] for r in te.ref]
    ld_te = np.log(te["mass_gsm"].to_numpy() / 1000.0)
    ld_tr = np.log(tr["mass_gsm"].to_numpy() / 1000.0)
    gt_dc = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    rows = []

    def score(name, dc_hat, kind):
        e = 100 * np.abs(dc_hat - gt_dc)
        r = np.corrcoef(dc_hat, gt_dc)[0, 1]
        rows.append({"model": name, "kind": kind, "DC_MAE_pct": e.mean(), "DC_MedAE_pct": np.median(e),
                     "DC_R2": 1 - ((dc_hat - gt_dc) ** 2).sum() / ((gt_dc - gt_dc.mean()) ** 2).sum(),
                     "DC_pearson": r})

    # published methods: real simulations and surrogate
    for m in ["MechMet", "MechIm"]:
        sim = pairs[pairs.source == m].set_index("ref")["drape_coefficient"].reindex(te.ref).to_numpy()
        ok = ~np.isnan(sim)
        if ok.all():
            score(f"EG2024-{m} (simulated)", sim, "published")
        else:
            sim[~ok] = np.nan
            tmp = gt_dc.copy()
            e = 100 * np.abs(sim[ok] - gt_dc[ok])
            rows.append({"model": f"EG2024-{m} (simulated)", "kind": "published", "DC_MAE_pct": e.mean(),
                         "DC_MedAE_pct": np.median(e), "DC_R2": np.nan, "DC_pearson": np.corrcoef(sim[ok], gt_dc[ok])[0, 1]})
        sc = pairs[pairs.source == m].set_index("ref")
        if ok.all():
            P = sc.loc[te.ref, [f"log_{t}" for t in TARGETS]].to_numpy()
            score(f"EG2024-{m} (surrogate)", dc_of(P, ld_te, models_te), "published")
    # GT parameters through the cross-fitted surrogate = surrogate floor
    Pg = pairs[pairs.source == "GT"].set_index("ref").loc[te.ref, [f"log_{t}" for t in TARGETS]].to_numpy()
    score("GT params (surrogate floor)", dc_of(Pg, ld_te, models_te), "reference")
    # A: parameter-first, all saved learners
    for p in sorted((RESULTS / "preds").glob("preds_*.npz")):
        name = p.stem.replace("preds_", "")
        score(name, dc_of(np.load(p)["test"], ld_te, models_te), "A parameter-first")
    # B: drape-first, DC label from surrogate on training labels
    dc_tr = full.predict(np.column_stack([Ytr, ld_tr]))
    groups = duplicate_groups(tr)
    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, random_state=SEED)
    krr = make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02))
    for nm, est in [("LightGBM", lgb), ("KRR", krr)]:
        score(f"{nm} drape-first", clone(est).fit(Xtr, dc_tr).predict(Xte), "B drape-first")
    # C: drape-weighted per-target LightGBM
    eps = 0.1
    W = np.zeros_like(Ytr)
    base_X = np.column_stack([Ytr, ld_tr])
    for j in range(6):
        e = np.zeros(7); e[j] = eps
        W[:, j] = ((full.predict(base_X + e) - full.predict(base_X - e)) / (2 * eps)) ** 2
    Wn = W / W.mean(axis=0, keepdims=True)
    for lam in [0.5, 1.0]:
        P = np.zeros((len(Xte), 6))
        for j in range(6):
            w = (1 - lam) + lam * Wn[:, j]
            P[:, j] = clone(lgb).fit(Xtr, Ytr[:, j], sample_weight=w).predict(Xte)
        score(f"LightGBM drape-weighted (lam={lam})", dc_of(P, ld_te, models_te), "C drape-weighted")
    df = pd.DataFrame(rows).sort_values("DC_MAE_pct")
    df.to_csv(RESULTS / "drape_space_eval.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
