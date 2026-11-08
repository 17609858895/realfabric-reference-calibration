"""Paired bootstrap comparisons and SHAP attributions. Matched reference-fabric predictions are compared within their implemented simulation/surrogate evaluation procedures. These source-specific evaluation identities are retained in the manuscript."""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from common import PROC, RESULTS, SHORT, TARGETS, engineer, load_raw, targets_log

warnings.filterwarnings("ignore")
ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
rng = np.random.default_rng(5)


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    pte = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["test"] for n in ENSEMBLE], axis=0)
    sp = pd.read_csv(PROC / "scene_params.csv")
    pub = sp[sp.source == "MechMet"].set_index("ref")[TARGETS].copy()
    pub[TARGETS[:3]] *= 1e8
    Ppub = np.log(pub.loc[te.ref].to_numpy())
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    gt_dc = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    pub_dc = pairs[pairs.source == "MechMet"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    # cross-fitted surrogate DC for our ensemble (re-use step 07 helper)
    from importlib import import_module
    s07 = import_module("07_drape_space_eval")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    cf = s07.crossfit_surrogates(pairs, sur["features"], sur["model"])
    ens_dc = s07.dc_of(pte, np.log(te["mass_gsm"].to_numpy() / 1000), [cf[r] for r in te.ref])

    def per_fabric(P, dc):
        e2 = ((P - Yte) ** 2).mean(1)
        ape = np.median(np.abs(np.exp(P) - np.exp(Yte)) / np.exp(Yte), axis=1)
        return e2, ape, np.abs(dc - gt_dc) * 100

    A, B = per_fabric(pte, ens_dc), per_fabric(Ppub, pub_dc)
    rows = []
    for k, nm in enumerate(["log-MSE (fabric mean)", "fabric median APE", "DC abs error (pct)"]):
        d = A[k] - B[k]
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        rows.append({"metric": nm, "ours_mean": A[k].mean(), "EG2024_MechMet_mean": B[k].mean(),
                     "diff_mean": d.mean(), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "p_one_sided": (boot >= 0).mean(), "win_rate": (d < 0).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "paired_bootstrap_vs_EG2024.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(df.to_string(index=False))

    # SHAP of LightGBM (one model per target), training data
    imp = {}
    for j, t in enumerate(TARGETS):
        m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1,
                          random_state=2024).fit(Xtr, Ytr[:, j])
        sv = shap.TreeExplainer(m).shap_values(Xtr)
        imp[SHORT[t]] = np.abs(sv).mean(0)
        np.save(RESULTS / f"shap_values_{t}.npy", sv)
    imp = pd.DataFrame(imp, index=Xtr.columns)
    imp["mean"] = imp.mean(1)
    imp.sort_values("mean", ascending=False).to_csv(RESULTS / "shap_mean_abs.csv")
    print(imp.sort_values("mean", ascending=False).head(12).round(3))


if __name__ == "__main__":
    main()
