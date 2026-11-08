"""Step 05 - learn a drape surrogate  DC = h(log k_1..6, log areal density)  from the 584 released Cusick scenes.

Each test fabric was simulated three times (GT, MechMet, MechIm parameters) on the same 30 cm / 18 cm Cusick
setup, so the scenes give 584 (parameter -> drape) pairs.  CV is grouped by fabric.  The surrogate is used only
to (i) score new parameter predictions in drape space and (ii) define a drape-sensitivity weight; published
methods are scored both with their real simulated drapes and with the surrogate, so the comparison stays fair.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib

from common import PROC, RESULTS, TARGETS

warnings.filterwarnings("ignore")


def load_pairs() -> pd.DataFrame:
    sp = pd.read_csv(PROC / "scene_params.csv")
    dr = pd.read_csv(PROC / "drape_metrics.csv")
    df = sp.merge(dr, on=["ref", "source"])
    df[TARGETS[:3]] = df[TARGETS[:3]] * 1e8
    for t in TARGETS:
        df[f"log_{t}"] = np.log(df[t])
    df["log_density"] = np.log(df["density"])
    return df


FEATS = [f"log_{t}" for t in TARGETS] + ["log_density"]


def surrogate_models():
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(FEATS))) + WhiteKernel(1e-3)
    return {
        "PolyRidge(d=2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-4, 2, 30))),
        "GPR-ARD": make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                                                            random_state=0)),
        "CatBoost": CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=4, verbose=0, random_seed=0),
    }


def main() -> None:
    df = load_pairs()
    X, y, g = df[FEATS].to_numpy(), df["drape_coefficient"].to_numpy(), df["ref"].to_numpy()
    # sanity: published-method drape error from the real simulations (paper reports DRPct MAE ~9.7 for MechMet)
    piv = df.pivot_table(index="ref", columns="source", values="drape_coefficient")
    for m in ["MechMet", "MechIm"]:
        ok = piv[[m, "GT"]].dropna()
        print(f"simulated DC MAE {m} vs GT: {100 * np.mean(np.abs(ok[m] - ok['GT'])):.2f} %  (n={len(ok)})")
    rows = []
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    best, best_mae = None, 1e9
    for name, est in surrogate_models().items():
        oof = np.zeros_like(y)
        for a, b in folds:
            oof[b] = clone(est).fit(X[a], y[a]).predict(X[b])
        mae = 100 * np.mean(np.abs(oof - y))
        r2 = 1 - ((oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"surrogate": name, "cv_R2": r2, "cv_MAE_DCpct": mae})
        print(f"{name:16s} grouped-CV R2={r2:.4f}  MAE={mae:.2f} DC%")
        if mae < best_mae:
            best, best_mae, best_oof = name, mae, oof
    pd.DataFrame(rows).to_csv(RESULTS / "drape_surrogate_cv.csv", index=False)
    df["DC_surrogate_oof"] = best_oof
    df.to_csv(RESULTS / "drape_surrogate_oof.csv", index=False)
    final = clone(surrogate_models()[best]).fit(X, y)
    joblib.dump({"model": final, "features": FEATS, "name": best}, RESULTS / "drape_surrogate.joblib")
    # finite-difference sensitivity of DC to each log-parameter on GT scenes
    gt = df[df.source == "GT"]
    Xg = gt[FEATS].to_numpy()
    sens = {}
    for j, f in enumerate(FEATS):
        e = np.zeros(len(FEATS)); e[j] = 0.1
        sens[f] = np.mean(np.abs(final.predict(Xg + e) - final.predict(Xg - e)) / 0.2)
    print("mean |dDC/dlog k|:", {k: round(v, 4) for k, v in sens.items()})
    pd.Series(sens).to_csv(RESULTS / "drape_sensitivity.csv")


if __name__ == "__main__":
    main()
