"""Exploratory label, physics-offset, monotonicity and stacking comparisons on the released collection. The reference collection was reused during model exploration."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, TARGETS, duplicate_groups, engineer, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
SEED = 2024


def cv_test(est, Xtr, Ytr, Xte, Yte, folds, fit_fn=None):
    oof = np.zeros_like(Ytr)
    for a, b in folds:
        m = clone(est).fit(Xtr.iloc[a], Ytr[a])
        oof[b] = m.predict(Xtr.iloc[b])
    m = clone(est).fit(Xtr, Ytr)
    pte = m.predict(Xte)
    return metrics_block(Ytr, oof)["R2log_mean"], metrics_block(Yte, pte), oof, pte


class PhysicsOffsetHybrid:
    """y = Ridge(physics descriptors) + LightGBM(all descriptors) on the ridge residual."""

    def __init__(self, phys_cols, lgb_params=None):
        self.phys_cols = phys_cols
        self.lgb_params = lgb_params or {}

    def get_params(self, deep=False):
        return {"phys_cols": self.phys_cols, "lgb_params": self.lgb_params}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, Y):
        self.lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[self.phys_cols], Y)
        R = Y - self.lin.predict(X[self.phys_cols])
        self.gb = MultiOutputRegressor(LGBMRegressor(**self.lgb_params), n_jobs=6).fit(X, R)
        return self

    def predict(self, X):
        return self.lin.predict(X[self.phys_cols]) + self.gb.predict(X)


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Yte = targets_log(te)
    groups = duplicate_groups(tr)
    Y_met, Y_im = targets_log(tr, "_MechMet"), targets_log(tr, "_MechIm")
    folds = list(GroupKFold(n_splits=5).split(Xtr, Y_met, groups))
    lgbp = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=4, random_state=SEED)
    lgb = MultiOutputRegressor(LGBMRegressor(**lgbp), n_jobs=6)
    krr = make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02))
    rows = []

    # P1 label source
    for lab, Y in [("MechMet", Y_met), ("MechIm", Y_im), ("logmean", 0.5 * (Y_met + Y_im))]:
        for nm, est in [("LightGBM", lgb), ("KRR", krr)]:
            cv, tm, _, _ = cv_test(est, Xtr, Y, Xte, Yte, folds)
            rows.append({"probe": "P1 label", "variant": f"{nm} | labels={lab}", "cv_R2log": cv,
                         "test_R2log": tm["R2log_mean"], "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"]})

    # P2 physics scaling exponents (OLS with structure + fibre-family controls), train labels and test GT
    ctrl = [c for c in Xtr.columns if c.startswith("s_") or c.startswith("f_")][:-1] + ["woven"]
    print("\nP2 empirical scaling exponents: log k = a*log t + b*log m + controls")
    sc_rows = []
    for nm, X, Y in [("train labels", Xtr, Y_met), ("test GT", Xte, Yte)]:
        D = sm.add_constant(X[["log_thickness", "log_mass"] + ctrl].drop(columns=[c for c in ctrl if X[c].std() == 0]))
        for j, t in enumerate(TARGETS):
            res = sm.OLS(Y[:, j], D).fit(cov_type="HC3")
            a, b = res.params["log_thickness"], res.params["log_mass"]
            sa, sb = res.bse["log_thickness"], res.bse["log_mass"]
            sc_rows.append({"data": nm, "target": t, "a_thickness": a, "se_a": sa, "b_mass": b, "se_b": sb,
                            "R2": res.rsquared})
            print(f"  {nm:12s} {t:13s} a(t)={a:5.2f}±{sa:4.2f}  b(m)={b:5.2f}±{sb:4.2f}  R2={res.rsquared:.3f}")
    pd.DataFrame(sc_rows).to_csv(RESULTS / "probe_scaling_exponents.csv", index=False)

    # P3 physics-offset hybrid
    phys = ["log_thickness", "log_mass", "log_bulk_density", "woven"] + [c for c in Xtr.columns if c.startswith("f_")]
    hyb = PhysicsOffsetHybrid(phys, lgbp)
    cv, tm, _, _ = cv_test(hyb, Xtr, Y_met, Xte, Yte, folds)
    rows.append({"probe": "P3 hybrid", "variant": "Ridge(physics)+LightGBM residual", "cv_R2log": cv,
                 "test_R2log": tm["R2log_mean"], "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"]})

    # P4 monotone constraints on thickness and mass
    mono = [0] * Xtr.shape[1]
    for c in ["log_thickness", "log_mass"]:
        mono[list(Xtr.columns).index(c)] = 1
    lgbm_mono = MultiOutputRegressor(LGBMRegressor(**lgbp, monotone_constraints=mono), n_jobs=6)
    cv, tm, _, _ = cv_test(lgbm_mono, Xtr, Y_met, Xte, Yte, folds)
    rows.append({"probe": "P4 monotone", "variant": "LightGBM monotone(t,m)", "cv_R2log": cv,
                 "test_R2log": tm["R2log_mean"], "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"]})

    # P5 stacking over saved 20-learner predictions
    names = [p.stem.replace("preds_", "") for p in sorted((RESULTS / "preds").glob("preds_*.npz"))]
    P = {n: np.load(RESULTS / "preds" / f"preds_{n}.npz") for n in names}
    # inner-CV to evaluate stacker on train OOF honestly: meta fit on folds of the OOF matrix
    for mode in ["per-target", "cross-target"]:
        oof_meta = np.zeros_like(Y_met)
        te_meta = np.zeros_like(Yte)
        for j in range(6):
            cols = [j] if mode == "per-target" else list(range(6))
            Z = np.column_stack([P[n]["oof"][:, cols] for n in names])
            Zt = np.column_stack([P[n]["test"][:, cols] for n in names])
            meta = RidgeCV(alphas=np.logspace(-2, 3, 30))
            for a, b in folds:
                oof_meta[b, j] = clone(meta).fit(Z[a], Y_met[a, j]).predict(Z[b])
            te_meta[:, j] = clone(meta).fit(Z, Y_met[:, j]).predict(Zt)
        tm = metrics_block(Yte, te_meta)
        rows.append({"probe": "P5 stacking", "variant": f"Ridge meta, {mode}", "cv_R2log": metrics_block(Y_met, oof_meta)["R2log_mean"],
                     "test_R2log": tm["R2log_mean"], "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"]})
    # simple average of top-k by CV
    cvs = {n: metrics_block(Y_met, P[n]["oof"])["R2log_mean"] for n in names}
    for k in [3, 5, 8, 20]:
        top = sorted(cvs, key=cvs.get, reverse=True)[:k]
        tm = metrics_block(Yte, np.mean([P[n]["test"] for n in top], axis=0))
        rows.append({"probe": "P5 averaging", "variant": f"mean of top-{k} by CV", "cv_R2log": metrics_block(Y_met, np.mean([P[n]['oof'] for n in top], axis=0))["R2log_mean"],
                     "test_R2log": tm["R2log_mean"], "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "probe_improvements.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
