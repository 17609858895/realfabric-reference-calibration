"""Metadata-distribution diagnostics and empirical reference-residual interval comparisons. A training-versus-reference classifier describes metadata separability. Anchor-based variants use residual quantiles and are evaluated on each draw's non-anchor remainder. Quantile construction alone is not a proof of coverage for every implemented variant."""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from common import PROC, RESULTS, TARGETS, duplicate_groups, engineer, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
ALPHA = 0.10
ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
rng = np.random.default_rng(7)


def cq(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    groups = duplicate_groups(tr)

    # (i) adversarial validation
    Xall = pd.concat([Xtr, Xte], ignore_index=True)
    lab = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    p = cross_val_predict(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1),
                          Xall, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(lab, p)
    print(f"adversarial validation AUC train-vs-test (metadata): {auc:.3f}")

    # ensemble point predictions and spread
    P = {n: np.load(RESULTS / "preds" / f"preds_{n}.npz") for n in ENSEMBLE}
    oof = np.mean([P[n]["oof"] for n in ENSEMBLE], axis=0)
    pte = np.mean([P[n]["test"] for n in ENSEMBLE], axis=0)
    allnames = [p.stem.replace("preds_", "") for p in (RESULTS / "preds").glob("preds_*.npz")]
    spread_tr = np.std([np.load(RESULTS / "preds" / f"preds_{n}.npz")["oof"] for n in allnames], axis=0)
    spread_te = np.std([np.load(RESULTS / "preds" / f"preds_{n}.npz")["test"] for n in allnames], axis=0)
    print("ensemble CV R2log", round(metrics_block(Ytr, oof)["R2log_mean"], 3), "test R2log",
          round(metrics_block(Yte, pte)["R2log_mean"], 3))
    R_tr, R_te = np.abs(Ytr - oof), np.abs(Yte - pte)
    print("mean |resid| train-OOF", R_tr.mean(0).round(3), " test", R_te.mean(0).round(3))
    print("spearman(spread, |test resid|):",
          [round(pd.Series(spread_te[:, j]).corr(pd.Series(R_te[:, j]), method="spearman"), 2) for j in range(6)])

    # local difficulty model sigma(x): predicts log|OOF residual| from metadata + ensemble spread (train only)
    sig_tr, sig_te = np.zeros_like(Ytr), np.zeros_like(Yte)
    folds = list(GroupKFold(5).split(Xtr, Ytr, groups))
    for j in range(6):
        Ztr = np.column_stack([Xtr.to_numpy(), spread_tr[:, j]])
        Zte = np.column_stack([Xte.to_numpy(), spread_te[:, j]])
        y = np.log(R_tr[:, j] + 1e-3)
        mdl = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, verbose=-1)
        for a, b in folds:
            sig_tr[b, j] = np.exp(mdl.fit(Ztr[a], y[a]).predict(Ztr[b]))
        sig_te[:, j] = np.exp(mdl.fit(Ztr, y).predict(Zte))
    print("spearman(sigma(x), |test resid|):",
          [round(pd.Series(sig_te[:, j]).corr(pd.Series(R_te[:, j]), method="spearman"), 2) for j in range(6)])

    # drape-space residuals
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")["model"]
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    ld_tr = np.log(tr["mass_gsm"].to_numpy() / 1000)
    ld_te = np.log(te["mass_gsm"].to_numpy() / 1000)
    dc_tr_lab = sur.predict(np.column_stack([Ytr, ld_tr]))
    dc_tr_hat = sur.predict(np.column_stack([oof, ld_tr]))
    dc_te_hat = sur.predict(np.column_stack([pte, ld_te]))
    dc_te_gt = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    Rdc_tr, Rdc_te = np.abs(dc_tr_lab - dc_tr_hat), np.abs(dc_te_gt - dc_te_hat)

    woven_te = te["woven"].to_numpy() == 1
    rows = []
    names = TARGETS + ["DC"]
    R_TE = np.column_stack([R_te, Rdc_te])
    R_TR = np.column_stack([R_tr, Rdc_tr])
    SIG_TE = np.column_stack([sig_te, np.ones(len(te))])

    # S0 internal calibration
    for j, nm in enumerate(names):
        q = cq(R_TR[:, j])
        cov = (R_TE[:, j] <= q).mean()
        rows.append({"setting": "S0 internal (released labels)", "target": nm, "n_cal": len(R_TR), "coverage": cov,
                     "width": 2 * q, "cov_woven": (R_TE[woven_te, j] <= q).mean(),
                     "cov_knit": (R_TE[~woven_te, j] <= q).mean()})

    n = len(te)
    for n_cal in [10, 20, 40, 80, 120]:
        acc = {s: {nm: [] for nm in names} for s in ["S1 anchor", "S2 PSAS", "S3 PSAS-Mondrian"]}
        for rep in range(300):
            idx = rng.permutation(n)
            c, e = idx[:n_cal], idx[n_cal:]
            for j, nm in enumerate(names):
                r, s = R_TE[:, j], SIG_TE[:, j]
                # S1 plain anchors
                q = cq(r[c])
                acc["S1 anchor"][nm].append(((r[e] <= q).mean(), 2 * q, (r[e][woven_te[e]] <= q).mean(),
                                             (r[e][~woven_te[e]] <= q).mean()))
                # S2 proxy-shaped anchor-scaled
                qs = cq(r[c] / s[c])
                acc["S2 PSAS"][nm].append(((r[e] <= qs * s[e]).mean(), 2 * np.mean(qs * s[e]),
                                           (r[e][woven_te[e]] <= qs * s[e][woven_te[e]]).mean(),
                                           (r[e][~woven_te[e]] <= qs * s[e][~woven_te[e]]).mean()))
                # S3 Mondrian by woven/knit
                cw, ck = c[woven_te[c]], c[~woven_te[c]]
                if len(cw) >= 5 and len(ck) >= 5:
                    qw, qk = cq(r[cw] / s[cw]), cq(r[ck] / s[ck])
                    qq = np.where(woven_te[e], qw, qk) * s[e]
                    acc["S3 PSAS-Mondrian"][nm].append(((r[e] <= qq).mean(), 2 * np.mean(qq),
                                                        (r[e][woven_te[e]] <= qq[woven_te[e]]).mean(),
                                                        (r[e][~woven_te[e]] <= qq[~woven_te[e]]).mean()))
        for s, d in acc.items():
            for nm, v in d.items():
                if not v:
                    continue
                v = np.array(v)
                v[~np.isfinite(v)] = np.nan
                rows.append({"setting": s, "target": nm, "n_cal": n_cal, "coverage": np.nanmean(v[:, 0]),
                             "width": np.nanmedian(v[:, 1]), "cov_woven": np.nanmean(v[:, 2]),
                             "cov_knit": np.nanmean(v[:, 3]), "cov_p05": np.nanpercentile(v[:, 0], 5)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "conformal_results.csv", index=False)
    summ = df.assign(gap=lambda d: (d.cov_woven - d.cov_knit).abs()).groupby(["setting", "n_cal"]).agg(
        coverage=("coverage", "mean"), width=("width", "mean"), woven_knit_gap=("gap", "mean")).reset_index()
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summ.to_string(index=False))
        print(df[(df.target == "DC")].to_string(index=False))
    pd.DataFrame({"auc_adversarial": [auc]}).to_csv(RESULTS / "adversarial_validation.csv", index=False)


if __name__ == "__main__":
    main()
