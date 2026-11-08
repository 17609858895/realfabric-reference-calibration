"""Retrospective small-reference correction of six stiffness predictions. Residual shifts, affine corrections, ridge-affine corrections and pooled offsets are fitted to released GT-scene reference anchors, then scored on the remaining reference fabrics. No new laboratory measurements are collected by this script."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, TARGETS, engineer, load_raw, metrics_block, targets_log

ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
rng = np.random.default_rng(11)


def eb_shift(res_c: np.ndarray) -> np.ndarray:
    """res_c: (n_cal, 6) anchor residuals -> empirical-Bayes per-target offsets pooled toward their grand mean."""
    n = len(res_c)
    m = res_c.mean(0)
    v = res_c.var(0, ddof=1) / n  # sampling variance of each mean
    g = m.mean()
    tau2 = max(np.var(m, ddof=1) - v.mean(), 0.0)
    w = tau2 / (tau2 + v)
    return g + w * (m - g)


def main() -> None:
    tr, te = load_raw()
    Yte = targets_log(te)
    P = {n: np.load(RESULTS / "preds" / f"preds_{n}.npz") for n in ENSEMBLE}
    pte = np.mean([P[n]["test"] for n in ENSEMBLE], axis=0)
    R = Yte - pte
    print("test mean residual (GT - pred), log units:", R.mean(0).round(3))
    print("residual correlation across targets:\n", np.corrcoef(R.T).round(2))
    woven = te["woven"].to_numpy() == 1
    print("mean residual woven:", R[woven].mean(0).round(3), " knit:", R[~woven].mean(0).round(3))
    print("by brand:", pd.DataFrame(R, columns=TARGETS).groupby(te["brand"].values).mean().round(2))
    n = len(te)
    rows = []
    for n_cal in [5, 10, 20, 40, 80]:
        acc = {k: [] for k in ["none", "shift", "affine", "ridge-affine", "pooled-EB", "pooled-EB+M"]}
        for rep in range(300):
            idx = rng.permutation(n)
            c, e = idx[:n_cal], idx[n_cal:]
            Yc, Pc, Ye, Pe = Yte[c], pte[c], Yte[e], pte[e]
            acc["none"].append(metrics_block(Ye, Pe))
            acc["shift"].append(metrics_block(Ye, Pe + (Yc - Pc).mean(0)))
            A = np.zeros_like(Pe)
            Arg = np.zeros_like(Pe)
            for j in range(6):
                X = np.column_stack([np.ones(n_cal), Pc[:, j]])
                b = np.linalg.lstsq(X, Yc[:, j], rcond=None)[0] if n_cal >= 3 else np.array([0, 1.0])
                A[:, j] = b[0] + b[1] * Pe[:, j]
                # ridge toward identity: minimise ||y - a - b*p||^2 + lam*(b-1)^2, lam = 20
                pc = Pc[:, j] - Pc[:, j].mean()
                lam = 20.0
                bb = (pc @ (Yc[:, j] - Yc[:, j].mean()) + lam) / (pc @ pc + lam)
                aa = Yc[:, j].mean() - bb * Pc[:, j].mean()
                Arg[:, j] = aa + bb * Pe[:, j]
            acc["affine"].append(metrics_block(Ye, A))
            acc["ridge-affine"].append(metrics_block(Ye, Arg))
            acc["pooled-EB"].append(metrics_block(Ye, Pe + eb_shift(Yc - Pc)))
            Pm = Pe.copy()
            ok = True
            for grp in [True, False]:
                cg = woven[c] == grp
                if cg.sum() < 3:
                    ok = False
                    break
                Pm[woven[e] == grp] += eb_shift((Yc - Pc)[cg])
            if ok:
                acc["pooled-EB+M"].append(metrics_block(Ye, Pm))
        for k, v in acc.items():
            if not v:
                continue
            rows.append({"n_cal": n_cal, "method": k,
                         "R2log": np.mean([d["R2log_mean"] for d in v]),
                         "R2log_p05": np.percentile([d["R2log_mean"] for d in v], 5),
                         "NMAE": np.mean([d["NMAE_mean"] for d in v]),
                         "MdAPE": np.mean([d["MdAPE_mean"] for d in v]),
                         "RMSElog": np.mean([d["RMSElog_mean"] for d in v])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "anchor_recalibration.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
