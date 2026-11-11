"""Budget-matched interval constructions around the pooled anchor correction. On the anchor draws of the ACDC analysis (same seed and permutation order), the full-anchor centre with leave-one-anchor-out radii is compared with split-conformal, jackknife+, Student-t and in-sample constructions that use the same reference fabrics. Scores are retrospective empirical coverage, interval width and point accuracy on the non-anchor remainder of each draw."""
from __future__ import annotations

import warnings
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
ALPHA = 0.10
rng = np.random.default_rng(21)
s07 = import_module("07_drape_space_eval")
s12 = import_module("12_acdc_pipeline")
eb_shift, cq = s12.eb_shift, s12.cq


def kth(a, k, axis=0):
    """k-th smallest (1-based) along axis; -inf if k < 1, +inf if k > n."""
    n = a.shape[axis]
    shape = tuple(np.delete(a.shape, axis))
    if k < 1:
        return np.full(shape, -np.inf)
    if k > n:
        return np.full(shape, np.inf)
    return np.take(np.sort(a, axis=axis), k - 1, axis=axis)


def fin_mean(x):
    x = np.asarray(x, dtype=float)
    f = np.isfinite(x)
    return x[f].mean() if f.any() else np.nan


def main() -> None:
    tr, te = load_raw()
    Yte = targets_log(te)
    P = {n: np.load(RESULTS / "preds" / f"preds_{n}.npz") for n in ENSEMBLE}
    pte = np.mean([P[n]["test"] for n in ENSEMBLE], axis=0)
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    cf = s07.crossfit_surrogates(pairs, sur["features"], sur["model"])
    models_te = [cf[r] for r in te.ref]
    ld_te = np.log(te["mass_gsm"].to_numpy() / 1000)
    gt_dc = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    base_dc = s07.dc_of(pte, ld_te, models_te)
    grad = np.zeros_like(pte)
    for j in range(6):
        e = np.zeros(6)
        e[j] = 0.05
        grad[:, j] = (s07.dc_of(pte + e, ld_te, models_te) - s07.dc_of(pte - e, ld_te, models_te)) / 0.1

    R = Yte - pte
    n = len(te)
    methods = ["Uncorrected conformal", "Split conformal", "Jackknife", "Jackknife+", "Student-t radii",
               "In-sample radii"]
    rows, draws = [], []
    for n_cal in [10, 20, 40, 80]:
        acc = {m: [] for m in methods}
        k_hi = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
        k_lo = int(np.floor(ALPHA * (n_cal + 1)))
        tq = stats.t.ppf(1 - ALPHA / 2, n_cal - 1)
        for rep in range(300):
            idx = rng.permutation(n)
            c, e = idx[:n_cal], idx[n_cal:]
            res_c = R[c]
            sh = eb_shift(res_c)
            sh_loo = np.array([eb_shift(np.delete(res_c, i, axis=0)) for i in range(n_cal)])
            loo = res_c - sh_loo  # leave-one-out log residuals, (n_cal, 6)
            dc_loo = gt_dc[c] - (base_dc[c] + np.einsum("ij,ij->i", grad[c], sh_loo))
            dce = base_dc[e] + grad[e] @ sh  # full-anchor corrected DC
            Re = R[e]  # evaluation residuals before any shift

            def score(name, centre_shift, lo_p, hi_p, dc_point, lo_d, hi_d):
                """lo_p/hi_p: (6,) bounds relative to the uncorrected prediction; lo_d/hi_d: (n_e,) DC bounds."""
                inside = (Re >= lo_p) & (Re <= hi_p)
                mm = metrics_block(Yte[e], pte[e] + centre_shift)
                wp = hi_p - lo_p
                wd = hi_d - lo_d
                acc[name].append({"R2log": mm["R2log_mean"], "MdAPE": mm["MdAPE_mean"],
                                  "DC_MAE": 100 * np.mean(np.abs(dc_point - gt_dc[e])),
                                  "cov_param": inside.mean(), "width_param": fin_mean(wp),
                                  "inf_param": np.mean(~np.isfinite(wp)),
                                  "cov_DC": np.mean((gt_dc[e] >= lo_d) & (gt_dc[e] <= hi_d)),
                                  "width_DC": 100 * fin_mean(wd), "inf_DC": np.mean(~np.isfinite(wd))})

            # uncorrected conformal: zero shift, anchors only calibrate
            q = np.array([cq(np.abs(res_c[:, j])) for j in range(6)])
            qd = cq(np.abs(gt_dc[c] - base_dc[c]))
            score("Uncorrected conformal", np.zeros(6), -q, q, base_dc[e], base_dc[e] - qd, base_dc[e] + qd)

            # split conformal: first half fits the pooled shift, second half calibrates
            h = n_cal // 2
            sh1 = eb_shift(res_c[:h])
            q = np.array([cq(np.abs(res_c[h:, j] - sh1[j])) for j in range(6)])
            dc1 = base_dc[e] + grad[e] @ sh1
            qd = cq(np.abs(gt_dc[c[h:]] - (base_dc[c[h:]] + grad[c[h:]] @ sh1)))
            score("Split conformal", sh1, sh1 - q, sh1 + q, dc1, dc1 - qd, dc1 + qd)

            # jackknife (primary)
            q = np.array([cq(np.abs(loo[:, j])) for j in range(6)])
            qd = cq(np.abs(dc_loo))
            score("Jackknife", sh, sh - q, sh + q, dce, dce - qd, dce + qd)

            # jackknife+
            lo_p = kth(sh_loo - np.abs(loo), k_lo, axis=0)
            hi_p = kth(sh_loo + np.abs(loo), k_hi, axis=0)
            M = base_dc[e][:, None] + grad[e] @ sh_loo.T  # (n_e, n_cal) leave-one-out DC predictions
            lo_d = kth(M - np.abs(dc_loo)[None, :], k_lo, axis=1)
            hi_d = kth(M + np.abs(dc_loo)[None, :], k_hi, axis=1)
            score("Jackknife+", sh, lo_p, hi_p, dce, lo_d, hi_d)

            # Student-t radii from leave-one-out residuals
            q = tq * np.sqrt(np.mean(loo ** 2, axis=0))
            qd = tq * np.sqrt(np.mean(dc_loo ** 2))
            score("Student-t radii", sh, sh - q, sh + q, dce, dce - qd, dce + qd)

            # in-sample radii (no leave-one-out)
            q = np.array([cq(np.abs(res_c[:, j] - sh[j])) for j in range(6)])
            qd = cq(np.abs(gt_dc[c] - (base_dc[c] + grad[c] @ sh)))
            score("In-sample radii", sh, sh - q, sh + q, dce, dce - qd, dce + qd)

        for m in methods:
            d = pd.DataFrame(acc[m])
            rows.append({"method": m, "n_cal": n_cal, **d.mean().to_dict(),
                         "cov_param_p05": d["cov_param"].quantile(0.05), "cov_DC_p05": d["cov_DC"].quantile(0.05),
                         "cov_param_sd": d["cov_param"].std(ddof=1), "cov_DC_sd": d["cov_DC"].std(ddof=1)})
            d.insert(0, "rep", np.arange(len(d)))
            d.insert(0, "n_cal", n_cal)
            d.insert(0, "method", m)
            draws.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "interval_baselines.csv", index=False)
    pd.concat(draws).to_csv(RESULTS / "interval_baselines_draws.csv", index=False)
    with pd.option_context("display.width", 260, "display.precision", 4, "display.max_columns", 30):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
