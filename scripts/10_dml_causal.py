"""Cross-fitted partially linear models of adjusted observational fabric associations. Exposure-specific controls, operational-group folds and group-clustered standard errors are used. Perturbations assess numerical stability; the coefficients summarize the available released collection."""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold

from common import RESULTS, TARGETS, duplicate_groups, engineer, load_raw, targets_log

warnings.filterwarnings("ignore")
rng = np.random.default_rng(3)
FAM = ["f_cellulose_natural", "f_cellulose_regen", "f_protein", "f_polyester", "f_polyamide", "f_other_synth"]


def nuis_reg():
    return LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=15,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)


def nuis_clf():
    return LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=15,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)


def plr(Y, T, X, groups, binary=False, reps=5, seed=0):
    thetas, ses = [], []
    for r in range(reps):
        # shuffle group ids so each repetition uses a different fold assignment
        perm = np.random.default_rng(seed + r).permutation(groups.max() + 1)
        g = perm[groups]
        Yr, Tr = np.zeros_like(Y), np.zeros_like(T, dtype=float)
        for a, b in GroupKFold(5).split(X, Y, g):
            Yr[b] = Y[b] - nuis_reg().fit(X[a], Y[a]).predict(X[b])
            if binary:
                e = nuis_clf().fit(X[a], T[a]).predict_proba(X[b])[:, 1]
                Tr[b] = T[b] - np.clip(e, 0.02, 0.98)
            else:
                Tr[b] = T[b] - nuis_reg().fit(X[a], T[a]).predict(X[b])
        th = (Tr @ Yr) / (Tr @ Tr)
        psi = (Yr - th * Tr) * Tr / np.mean(Tr ** 2)
        cl = pd.Series(psi).groupby(groups).sum().to_numpy()
        se = np.sqrt((cl ** 2).sum()) / len(Y)
        thetas.append(th)
        ses.append(se)
    i = np.argsort(thetas)[len(thetas) // 2]
    return float(np.median(thetas)), float(ses[i])


def design(Xe: pd.DataFrame, lever: str):
    base = Xe[FAM].idxmax(axis=1)
    base_oh = pd.get_dummies(base, prefix="base").astype(float)
    S = Xe[[c for c in Xe.columns if c.startswith("s_")]]
    if lever == "elastane_10pp":
        T = Xe["f_elastane"].to_numpy() * 10
        X = pd.concat([base_oh, S, Xe[["woven", "log_thickness", "log_mass"]]], axis=1)
        return T, X, False
    if lever == "woven":
        T = Xe["woven"].to_numpy()
        X = Xe[FAM + ["f_elastane", "log_thickness", "log_mass"]]
        return T, X, True
    if lever == "log_thickness":
        T = Xe["log_thickness"].to_numpy()
        X = pd.concat([S, Xe[["woven", "log_mass", "f_elastane"] + FAM]], axis=1)
        return T, X, False
    if lever == "log_mass":
        T = Xe["log_mass"].to_numpy()
        X = pd.concat([S, Xe[["woven", "log_thickness", "f_elastane"] + FAM]], axis=1)
        return T, X, False
    raise ValueError(lever)


def load_all():
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    gtr = duplicate_groups(tr)
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")["model"]
    dc_tr = sur.predict(np.column_stack([Ytr, np.log(tr["mass_gsm"] / 1000)]))
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    dc_te = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    outcomes_tr = {t: Ytr[:, j] for j, t in enumerate(TARGETS)} | {"DC": dc_tr}
    outcomes_te = {t: Yte[:, j] for j, t in enumerate(TARGETS)} | {"DC": dc_te}
    return Xtr, Xte, gtr, outcomes_tr, outcomes_te


def one_effect(lever: str, oc: str, seed: int) -> dict:
    Xtr, Xte, gtr, outcomes_tr, outcomes_te = load_all()
    r = np.random.default_rng(seed)
    T, X, binary = design(Xtr, lever)
    Tt, Xt, _ = design(Xte, lever)
    Xt = Xt.reindex(columns=X.columns, fill_value=0.0)
    Y = outcomes_tr[oc]
    th, se = plr(Y, T, X.to_numpy(), gtr, binary)
    th_pl, _ = plr(Y, r.permutation(T), X.to_numpy(), gtr, binary, reps=1)
    th_rc, _ = plr(Y, T, np.column_stack([X.to_numpy(), r.normal(size=len(Y))]), gtr, binary, reps=1)
    sub = []
    ug = np.unique(gtr)
    for _ in range(5):
        keep = np.isin(gtr, r.choice(ug, int(0.8 * len(ug)), replace=False))
        sub.append(plr(Y[keep], T[keep], X.to_numpy()[keep], gtr[keep], binary, reps=1)[0])
    th_te, se_te = plr(outcomes_te[oc], Tt, Xt.to_numpy(), np.arange(len(Tt)), binary, reps=3)
    return {"lever": lever, "outcome": oc, "theta": th, "se": se, "ci_lo": th - 1.96 * se, "ci_hi": th + 1.96 * se,
            "placebo": th_pl, "random_cc": th_rc, "subset_mean": np.mean(sub), "subset_sd": np.std(sub),
            "test_theta": th_te, "test_se": se_te, "sign_agree": bool(np.sign(th) == np.sign(th_te))}


def one_het(grp: str, oc: str) -> dict:
    Xtr, _, gtr, outcomes_tr, _ = load_all()
    mask = Xtr["woven"].to_numpy() == (1 if grp == "woven" else 0)
    T, X, _ = design(Xtr[mask].reset_index(drop=True), "elastane_10pp")
    th, se = plr(outcomes_tr[oc][mask], T, X.to_numpy(), gtr[mask], False, reps=3)
    return {"subgroup": grp, "outcome": oc, "theta": th, "se": se}


def main() -> None:
    from joblib import Parallel, delayed
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    Xtr, _, gtr, _, _ = load_all()
    T, X, _ = design(Xtr, "woven")
    e = np.zeros(len(T))
    for a, b in GroupKFold(5).split(X, T, gtr):
        e[b] = nuis_clf().fit(X.to_numpy()[a], T[a]).predict_proba(X.to_numpy()[b])[:, 1]
    print(f"woven propensity range {e.min():.3f}-{e.max():.3f}, share in [0.05,0.95] "
          f"{np.mean((e > .05) & (e < .95)):.2f}", flush=True)
    pd.DataFrame({"propensity_woven": e, "woven": T}).to_csv(RESULTS / "dml_propensity_woven.csv", index=False)

    jobs = [(lv, oc) for lv in ["elastane_10pp", "woven", "log_thickness", "log_mass"] for oc in TARGETS + ["DC"]]
    rows = Parallel(n_jobs=28, verbose=0)(delayed(one_effect)(lv, oc, 100 + i) for i, (lv, oc) in enumerate(jobs))
    df = pd.DataFrame(rows)
    df["p"] = 2 * norm.sf(np.abs(df["theta"] / df["se"]))
    df["q_bh"] = multipletests(df["p"], method="fdr_bh")[1]
    df.to_csv(RESULTS / "dml_effects.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 20):
        print(df.drop(columns=["ci_lo", "ci_hi"]).to_string(index=False), flush=True)

    het = Parallel(n_jobs=10)(delayed(one_het)(g, oc) for g in ["knit", "woven"]
                              for oc in ["kStretchWarp", "kStretchWeft", "kStretchBias", "kBendingBias", "DC"])
    het = pd.DataFrame(het)
    het.to_csv(RESULTS / "dml_elastane_heterogeneity.csv", index=False)
    with pd.option_context("display.precision", 3):
        print(het.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
