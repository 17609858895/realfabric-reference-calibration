"""Anchor-calibrated drape correction (ACDC): pooled residual offsets and empirical residual intervals. Leave-one-anchor-out scores are combined with a centre estimated from the full anchor set. The code evaluates retrospective coverage; it does not implement the exact jackknife+ interval construction. A0 uses released-label OOF residuals; A1 adds reference residual scales; A2 adds separate shifts; A3 pools shifts; A4 uses construction strata with a minimum-five marginal fallback."""
from __future__ import annotations

import warnings
from importlib import import_module

import joblib
import numpy as np
import pandas as pd

from common import RESULTS, TARGETS, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
ALPHA = 0.10
rng = np.random.default_rng(21)
s07 = import_module("07_drape_space_eval")


def cq(s, alpha=ALPHA):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s)[k - 1]


def eb_shift(res):
    n = len(res)
    m = res.mean(0)
    if n < 3:
        return np.full(res.shape[1], m.mean())
    v = res.var(0, ddof=1) / n
    g = m.mean()
    tau2 = max(np.var(m, ddof=1) - v.mean(), 0.0)
    return g + (tau2 / (tau2 + v)) * (m - g)


def mean_shift(res):
    return res.mean(0)


def main() -> None:
    tr, te = load_raw()
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    P = {n: np.load(RESULTS / "preds" / f"preds_{n}.npz") for n in ENSEMBLE}
    oof = np.mean([P[n]["oof"] for n in ENSEMBLE], axis=0)
    pte = np.mean([P[n]["test"] for n in ENSEMBLE], axis=0)
    woven = te["woven"].to_numpy() == 1
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    cf = s07.crossfit_surrogates(pairs, sur["features"], sur["model"])
    models_te = [cf[r] for r in te.ref]
    ld_te = np.log(te["mass_gsm"].to_numpy() / 1000)
    gt_dc = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    ld_tr = np.log(tr["mass_gsm"].to_numpy() / 1000)
    dc_tr_lab = sur["model"].predict(np.column_stack([Ytr, ld_tr]))
    dc_tr_hat = sur["model"].predict(np.column_stack([oof, ld_tr]))

    # surrogate DC is (nearly) linear in small parameter shifts; precompute per-fabric gradient for speed
    base_dc = s07.dc_of(pte, ld_te, models_te)
    grad = np.zeros_like(pte)
    for j in range(6):
        e = np.zeros(6); e[j] = 0.05
        grad[:, j] = (s07.dc_of(pte + e, ld_te, models_te) - s07.dc_of(pte - e, ld_te, models_te)) / 0.1

    def dc_shifted(idx, shift):  # first-order DC of shifted parameters
        return base_dc[idx] + grad[idx] @ shift

    n = len(te)
    rows = []
    # A0 internal
    qp = np.array([cq(np.abs(Ytr[:, j] - oof[:, j])) for j in range(6)])
    qd = cq(np.abs(dc_tr_lab - dc_tr_hat))
    m = metrics_block(Yte, pte)
    rows.append({"variant": "A0 internal", "n_cal": 0, "R2log": m["R2log_mean"], "MdAPE": m["MdAPE_mean"],
                 "NMAE": m["NMAE_mean"], "DC_MAE": 100 * np.mean(np.abs(base_dc - gt_dc)),
                 "cov_param": np.mean(np.abs(Yte - pte) <= qp), "width_param": 2 * qp.mean(),
                 "cov_DC": np.mean(np.abs(gt_dc - base_dc) <= qd), "width_DC": 2 * qd,
                 "cov_gap_wk": abs(np.mean(np.abs(Yte - pte)[woven] <= qp) - np.mean(np.abs(Yte - pte)[~woven] <= qp))})

    variants = {"A1 +anchors": (None, False), "A2 +shift": (mean_shift, False), "A3 +pooling": (eb_shift, False),
                "A4 +Mondrian (ACDC)": (eb_shift, True)}
    for n_cal in [10, 20, 40, 80]:
        acc = {v: [] for v in variants}
        for rep in range(300):
            idx = rng.permutation(n)
            c, e = idx[:n_cal], idx[n_cal:]
            res_c = Yte[c] - pte[c]
            for vname, (shift_fn, mondrian) in variants.items():
                if shift_fn is None:
                    sh = np.zeros(6)
                    loo = res_c.copy()
                else:
                    sh = shift_fn(res_c)
                    loo = np.array([res_c[i] - shift_fn(np.delete(res_c, i, axis=0)) for i in range(n_cal)])
                Pe = pte[e] + sh
                dce = dc_shifted(e, sh)
                # DC residuals at anchors (LOO shift)
                dcr = np.array([gt_dc[c[i]] - dc_shifted(c[i:i + 1], (np.zeros(6) if shift_fn is None else
                                shift_fn(np.delete(res_c, i, axis=0))))[0] for i in range(n_cal)])
                if mondrian and min(woven[c].sum(), (~woven[c]).sum()) >= 5:
                    qp_w = np.array([cq(np.abs(loo[woven[c], j])) for j in range(6)])
                    qp_k = np.array([cq(np.abs(loo[~woven[c], j])) for j in range(6)])
                    QP = np.where(woven[e][:, None], qp_w, qp_k)
                    QD = np.where(woven[e], cq(np.abs(dcr[woven[c]])), cq(np.abs(dcr[~woven[c]])))
                else:
                    QP = np.tile([cq(np.abs(loo[:, j])) for j in range(6)], (len(e), 1))
                    QD = np.full(len(e), cq(np.abs(dcr)))
                inside = np.abs(Yte[e] - Pe) <= QP
                mm = metrics_block(Yte[e], Pe)
                acc[vname].append({"R2log": mm["R2log_mean"], "MdAPE": mm["MdAPE_mean"], "NMAE": mm["NMAE_mean"],
                                   "DC_MAE": 100 * np.mean(np.abs(dce - gt_dc[e])),
                                   "cov_param": inside.mean(), "width_param": 2 * np.mean(QP[np.isfinite(QP)]) if np.isfinite(QP).any() else np.nan,
                                   "cov_DC": np.mean(np.abs(gt_dc[e] - dce) <= QD),
                                   "width_DC": 2 * np.mean(QD[np.isfinite(QD)]) if np.isfinite(QD).any() else np.nan,
                                   "cov_gap_wk": abs(inside[woven[e]].mean() - inside[~woven[e]].mean())})
        for vname, v in acc.items():
            d = pd.DataFrame(v).replace([np.inf, -np.inf], np.nan)
            rows.append({"variant": vname, "n_cal": n_cal, **d.mean().to_dict(),
                         "cov_param_p05": d["cov_param"].quantile(0.05), "cov_DC_p05": d["cov_DC"].quantile(0.05)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "acdc_ablation.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
