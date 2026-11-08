"""Compare linearized and nonlinear drape scoring on matched anchor draws.

This is a sensitivity analysis of the existing surrogate, not a new simulator run.
Stiffness models and their predictions are unchanged.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json

import joblib
import numpy as np
import pandas as pd

from common import RESULTS, load_raw, targets_log

s07 = import_module("07_drape_space_eval")
s12 = import_module("12_acdc_pipeline")


def grouped_predict(X, models):
    out = np.empty(len(X))
    for model in dict.fromkeys(models):
        ix = np.array([m is model for m in models])
        out[ix] = model.predict(X[ix])
    return out


def main(reps=100):
    _, te = load_raw()
    Y = targets_log(te)
    names = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
    p = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["test"] for n in names], axis=0)
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    saved = joblib.load(RESULTS / "drape_surrogate.joblib")
    print("Fitting five cross-fitted surrogate scorers", flush=True)
    cf = s07.crossfit_surrogates(pairs, saved["features"], saved["model"])
    models = [cf[r] for r in te.ref]
    ld = np.log(te.mass_gsm.to_numpy() / 1000)
    gt = pairs[pairs.source == "GT"].set_index("ref").drape_coefficient.loc[te.ref].to_numpy()
    base = grouped_predict(np.column_stack([p, ld]), models)
    grad = np.empty_like(p)
    for j in range(6):
        step = np.zeros(6); step[j] = 0.05
        grad[:, j] = (grouped_predict(np.column_stack([p + step, ld]), models)
                      - grouped_predict(np.column_stack([p - step, ld]), models)) / 0.1
    rng = np.random.default_rng(216)
    rows = []
    for rep in range(reps):
        idx = rng.permutation(len(te)); c, e = idx[:20], idx[20:]
        res = Y[c] - p[c]
        shift = s12.eb_shift(res)
        loo_shifts = np.array([s12.eb_shift(np.delete(res, i, axis=0)) for i in range(20)])
        exact_e = grouped_predict(np.column_stack([p[e] + shift, ld[e]]), [models[i] for i in e])
        exact_c = grouped_predict(np.column_stack([p[c] + loo_shifts, ld[c]]), [models[i] for i in c])
        lin_e = base[e] + grad[e] @ shift
        lin_c = base[c] + np.einsum("ij,ij->i", grad[c], loo_shifts)
        q_exact = s12.cq(np.abs(gt[c] - exact_c))
        q_lin = s12.cq(np.abs(gt[c] - lin_c))
        rows.append({"rep": rep, "n_cal": 20,
                     "DC_MAE_linear_pts": 100*np.mean(np.abs(lin_e-gt[e])),
                     "DC_MAE_nonlinear_pts": 100*np.mean(np.abs(exact_e-gt[e])),
                     "coverage_linear": np.mean(np.abs(lin_e-gt[e]) <= q_lin),
                     "coverage_nonlinear": np.mean(np.abs(exact_e-gt[e]) <= q_exact),
                     "width_linear_pts": 200*q_lin, "width_nonlinear_pts": 200*q_exact,
                     "approx_MAE_pts": 100*np.mean(np.abs(lin_e-exact_e)),
                     "approx_max_pts": 100*np.max(np.abs(lin_e-exact_e)),
                     "outside_01_nonlinear": np.mean((exact_e < 0) | (exact_e > 1))})
        if (rep+1) % 10 == 0: print(f"Draws {rep+1}/{reps}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "nonlinear_dc_sensitivity.csv", index=False)
    summary = {"repetitions": reps, "anchors": 20, "seed": 216,
               "means": df.drop(columns=["rep", "n_cal"]).mean().to_dict(),
               "paired_MAE_difference_pts_p05_p95": np.quantile(df.DC_MAE_nonlinear_pts-df.DC_MAE_linear_pts,[.05,.95]).tolist(),
               "approx_error_pts_p05_p95": np.quantile(df.approx_MAE_pts,[.05,.95]).tolist()}
    (RESULTS / "nonlinear_dc_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=100)
    main(ap.parse_args().reps)
