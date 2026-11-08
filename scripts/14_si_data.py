"""Supplementary numerical analyses: random versus operational-group folds, fabric-excluding surrogate scores and ten additional fold-split repeats for adjusted association estimates."""
from __future__ import annotations

import warnings
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, TARGETS, duplicate_groups, engineer, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
FD = RESULTS / "figdata"
s07 = import_module("07_drape_space_eval")
s10 = import_module("10_dml_causal")


def s1_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "KernelRidge": make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02)),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5, n_jobs=8,
                                              random_state=2024),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.6, n_jobs=8,
                                          random_state=2024),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                                                       min_child_samples=10, subsample=0.8, subsample_freq=1,
                                                       colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=2,
                                                       random_state=2024), n_jobs=3),
    }


def s1(Xtr, Ytr, groups):
    rows = []
    splits = {"Random": list(KFold(5, shuffle=True, random_state=0).split(Xtr)),
              "Grouped": list(GroupKFold(5).split(Xtr, Ytr, groups))}
    for name, est in s1_models().items():
        for scheme, folds in splits.items():
            oof = np.zeros_like(Ytr)
            for a, b in folds:
                oof[b] = clone(est).fit(Xtr.iloc[a], Ytr[a]).predict(Xtr.iloc[b])
            m = metrics_block(Ytr, oof)
            rows += [{"model": name, "scheme": scheme, "target": t, "R2log": m[f"R2log_{t}"]} for t in TARGETS]
            rows.append({"model": name, "scheme": scheme, "target": "mean", "R2log": m["R2log_mean"]})
        print(name, "done", flush=True)
    pd.DataFrame(rows).to_csv(FD / "SI1_random_vs_grouped.csv", index=False)
    sizes = pd.Series(groups).value_counts()
    sizes.value_counts().sort_index().rename_axis("group_size").rename("n_groups").to_csv(FD / "SI1_group_sizes.csv")


def s6(te):
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    cf = s07.crossfit_surrogates(pairs, sur["features"], sur["model"])
    gt = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"]
    rows = []
    for src in ["MechMet", "MechIm"]:
        q = pairs[pairs.source == src].set_index("ref")
        refs = [r for r in te.ref if r in q.index]
        P = q.loc[refs, [f"log_{t}" for t in TARGETS]].to_numpy()
        ld = q.loc[refs, "log_density"].to_numpy()
        dc_s = s07.dc_of(P, ld, [cf[r] for r in refs])
        for r, a, b in zip(refs, dc_s, q.loc[refs, "drape_coefficient"]):
            rows.append({"ref": r, "source": src, "dc_surrogate": a, "dc_simulated": b, "dc_gt": gt.loc[r]})
    pd.DataFrame(rows).to_csv(FD / "SI6_surrogate_scoring.csv", index=False)


def _rep(lever, oc, seed):
    Xtr, Xte, gtr, outcomes_tr, _ = s10.load_all()
    T, X, binary = s10.design(Xtr, lever)
    th, se = s10.plr(outcomes_tr[oc], T, X.to_numpy(), gtr, binary, reps=1, seed=seed)
    return {"lever": lever, "outcome": oc, "rep": seed, "theta": th, "se": se}


def s7():
    jobs = [(lv, oc, s) for lv in ["elastane_10pp", "woven", "log_thickness", "log_mass"]
            for oc in TARGETS + ["DC"] for s in range(1000, 1010)]
    rows = Parallel(n_jobs=28)(delayed(_rep)(*j) for j in jobs)
    pd.DataFrame(rows).to_csv(FD / "SI7_dml_repeats.csv", index=False)


def main():
    tr, te = load_raw()
    Xtr, Ytr = engineer(tr), targets_log(tr, "_MechMet")
    groups = duplicate_groups(tr)
    print("== S6"); s6(te)
    print("== S1"); s1(Xtr, Ytr, groups)
    print("== S7"); s7()


if __name__ == "__main__":
    main()
