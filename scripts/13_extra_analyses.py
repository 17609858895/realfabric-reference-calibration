"""Additional study analyses: operational groups, structure-family holdout, anchor policies, normalized bending associations, metadata discrimination, calibration draws, overlap subsets, partial residuals and ALE. Different validation tasks retain their implemented ensemble definitions. All results concern released labels, reference scenes and simulated drapes."""
from __future__ import annotations

import io
import json
import warnings
import zipfile
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (FIBRE_FAMILY, PROC, RESULTS, ROOT, TARGETS, duplicate_groups, engineer, load_raw,
                    metrics_block, targets_log)

warnings.filterwarnings("ignore")
FD = RESULTS / "figdata"
FD.mkdir(exist_ok=True)
ENSEMBLE = ["CatBoost", "LightGBM", "HistGB", "KernelRidge", "GPR", "ExtraTrees"]
s07 = import_module("07_drape_space_eval")
s10 = import_module("10_dml_causal")
s12 = import_module("12_acdc_pipeline")


def base_models():
    lgb = MultiOutputRegressor(LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                             verbose=-1, n_jobs=2, random_state=2024), n_jobs=3)
    et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.6, n_jobs=8, random_state=2024)
    krr = make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02))
    return [lgb, et, krr]


# ---------------------------------------------------------------- A
def tiers(tr, te, Xtr, Xte, Ytr, Yte):
    fam = tr["structure_name"].to_numpy()
    rows = []
    # tier 1: fabric-level grouped CV (6-model ensemble OOF from step 02)
    oof = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["oof"] for n in ENSEMBLE], axis=0)
    pte = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["test"] for n in ENSEMBLE], axis=0)
    # tier 2: leave-one-structure-family-out (3-model mean, all families incl. rare ones)
    lofo = np.zeros_like(Ytr)
    for f in np.unique(fam):
        b = fam == f
        lofo[b] = np.mean([clone(m).fit(Xtr[~b], Ytr[~b]).predict(Xtr[b]) for m in base_models()], axis=0)
    np.save(FD / "lofo_oof.npy", lofo)
    for tier, Y, P in [("Fabric", Ytr, oof), ("Family", Ytr, lofo), ("Pipeline", Yte, pte)]:
        m = metrics_block(Y, P)
        rows += [{"tier": tier, "target": t, "R2log": m[f"R2log_{t}"], "MdAPE": m[f"MdAPE_{t}"]} for t in TARGETS]
        rows.append({"tier": tier, "target": "mean", "R2log": m["R2log_mean"], "MdAPE": m["MdAPE_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(FD / "A_generalisation_tiers.csv", index=False)
    # per-family LOFO error
    pf = pd.DataFrame({"family": fam, "abs_err": np.abs(lofo - Ytr).mean(1), "abs_err_cv": np.abs(oof - Ytr).mean(1)})
    pf.groupby("family").agg(n=("abs_err", "size"), lofo=("abs_err", "mean"), cv=("abs_err_cv", "mean")).reset_index()\
        .to_csv(FD / "A_family_errors.csv", index=False)
    print(df.pivot(index="target", columns="tier", values="R2log").round(3))


# ---------------------------------------------------------------- B
def kcenter(Z, k, rng):
    idx = [int(rng.integers(len(Z)))]
    d = np.linalg.norm(Z - Z[idx[0]], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(idx)


def stratified(fam, k, rng):
    order = rng.permutation(len(fam))
    counts = pd.Series(fam).value_counts()
    quota = np.maximum(1, np.round(counts / counts.sum() * k)).astype(int)
    out = []
    for f in counts.index:
        pool = [i for i in order if fam[i] == f][: quota[f]]
        out += pool
    out = out[:k]
    rest = [i for i in order if i not in out]
    return np.array(out + rest[: k - len(out)])


def anchor_policies(te, Xte, Yte, pte, base_dc, grad, gt_dc):
    Z = StandardScaler().fit_transform(Xte[["woven", "log_thickness", "log_mass", "log_bulk_density"] +
                                           [c for c in Xte.columns if c.startswith("f_")]])
    fam = te["structure_name"].to_numpy()
    rng = np.random.default_rng(99)
    rows = []
    for n_cal in [10, 20, 40]:
        for pol in ["Random", "Diversity (k-center)", "Stratified"]:
            for rep in range(200):
                if pol == "Random":
                    c = rng.permutation(len(te))[:n_cal]
                elif pol.startswith("Diversity"):
                    c = kcenter(Z, n_cal, rng)
                else:
                    c = stratified(fam, n_cal, rng)
                e = np.setdiff1d(np.arange(len(te)), c)
                res = Yte[c] - pte[c]
                sh = s12.eb_shift(res)
                loo = np.array([res[i] - s12.eb_shift(np.delete(res, i, 0)) for i in range(n_cal)])
                q = np.array([s12.cq(np.abs(loo[:, j])) for j in range(6)])
                Pe = pte[e] + sh
                dce = base_dc[e] + grad[e] @ sh
                m = metrics_block(Yte[e], Pe)
                rows.append({"n_cal": n_cal, "policy": pol, "rep": rep, "R2log": m["R2log_mean"],
                             "MdAPE": m["MdAPE_mean"], "DC_MAE": 100 * np.mean(np.abs(dce - gt_dc[e])),
                             "coverage": np.mean(np.abs(Yte[e] - Pe) <= q), "width": 2 * np.mean(q)})
    df = pd.DataFrame(rows)
    df.to_csv(FD / "B_anchor_policies.csv", index=False)
    print(df.groupby(["n_cal", "policy"])[["R2log", "MdAPE", "DC_MAE", "coverage", "width"]].mean().round(3))


# ---------------------------------------------------------------- C
def peirce(tr, te, Xtr, Ytr, gtr):
    rows = []
    for lever in ["log_mass", "log_thickness"]:
        T, X, binary = s10.design(Xtr, lever)
        for j, t in enumerate(TARGETS[:3]):
            Y = Ytr[:, j] - Xtr["log_mass"].to_numpy()  # log(B/W), W proportional to areal mass
            th, se = s10.plr(Y, T, X.to_numpy(), gtr, binary, reps=5)
            rows.append({"lever": lever, "outcome": f"log(B/W) {t}", "theta": th, "se": se})
    df = pd.DataFrame(rows)
    df.to_csv(FD / "C_peirce_invariance.csv", index=False)
    print(df.round(3))
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    pairs["log_bending_length"] = (np.log(pairs[["kBendingWarp", "kBendingWeft", "kBendingBias"]]).mean(1)
                                   - pairs["log_density"]) / 3.0
    pairs[["ref", "source", "log_bending_length", "drape_coefficient", "log_density"]].to_csv(
        FD / "C_scene_bending_length.csv", index=False)


# ---------------------------------------------------------------- D
def roc(Xtr, Xte):
    Xall = pd.concat([Xtr, Xte], ignore_index=True)
    lab = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    p = cross_val_predict(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1),
                          Xall, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(lab, p)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(FD / "D_adversarial_roc.csv", index=False)
    print("AUC", roc_auc_score(lab, p))


# ---------------------------------------------------------------- E
def acdc_draws(te, Yte, pte, base_dc, grad, gt_dc, Ytr, oof, tr):
    rng = np.random.default_rng(21)
    rows = []
    for n_cal in [5, 10, 15, 20, 30, 40, 60, 80, 120]:
        for rep in range(300):
            c = rng.permutation(len(te))[:n_cal]
            e = np.setdiff1d(np.arange(len(te)), c)
            res = Yte[c] - pte[c]
            for vname, fn in [("A1", None), ("A3", s12.eb_shift)]:
                sh = np.zeros(6) if fn is None else fn(res)
                loo = res.copy() if fn is None else np.array([res[i] - fn(np.delete(res, i, 0)) for i in range(n_cal)])
                q = np.array([s12.cq(np.abs(loo[:, j])) for j in range(6)])
                dcr = np.array([gt_dc[c[i]] - (base_dc[c[i]] + grad[c[i]] @ (np.zeros(6) if fn is None else
                                fn(np.delete(res, i, 0)))) for i in range(n_cal)])
                qd = s12.cq(np.abs(dcr))
                Pe = pte[e] + sh
                dce = base_dc[e] + grad[e] @ sh
                rows.append({"n_cal": n_cal, "variant": vname, "rep": rep,
                             "cov_param": np.mean(np.abs(Yte[e] - Pe) <= q) if np.isfinite(q).all() else np.nan,
                             "cov_DC": np.mean(np.abs(gt_dc[e] - dce) <= qd) if np.isfinite(qd) else np.nan,
                             "width_param": 2 * np.mean(q) if np.isfinite(q).all() else np.nan,
                             "width_DC": 2 * qd if np.isfinite(qd) else np.nan,
                             "R2log": metrics_block(Yte[e], Pe)["R2log_mean"],
                             "DC_MAE": 100 * np.mean(np.abs(dce - gt_dc[e]))})
    pd.DataFrame(rows).to_csv(FD / "E_acdc_draws.csv", index=False)
    # internal (A0) reference values
    ld_tr = np.log(tr["mass_gsm"].to_numpy() / 1000)
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")["model"]
    qp = np.array([s12.cq(np.abs(Ytr[:, j] - oof[:, j])) for j in range(6)])
    qd = s12.cq(np.abs(sur.predict(np.column_stack([Ytr, ld_tr])) - sur.predict(np.column_stack([oof, ld_tr]))))
    a0 = {"cov_param": float(np.mean(np.abs(Yte - pte) <= qp)), "cov_DC": float(np.mean(np.abs(gt_dc - base_dc) <= qd)),
          "width_param": float(2 * qp.mean()), "width_DC": float(2 * qd)}
    json.dump(a0, open(FD / "E_acdc_internal.json", "w"), indent=2)
    # illustrative draw, n_cal = 20
    r2 = np.random.default_rng(2026)
    c = r2.permutation(len(te))[:20]
    e = np.setdiff1d(np.arange(len(te)), c)
    res = Yte[c] - pte[c]
    sh = s12.eb_shift(res)
    dcr = np.array([gt_dc[c[i]] - (base_dc[c[i]] + grad[c[i]] @ s12.eb_shift(np.delete(res, i, 0))) for i in range(20)])
    qd = s12.cq(np.abs(dcr))
    ex = pd.DataFrame({"ref": te.ref.to_numpy()[e], "woven": te.woven.to_numpy()[e], "gt_dc": gt_dc[e],
                       "dc_raw": base_dc[e], "dc_acdc": base_dc[e] + grad[e] @ sh, "q_dc": qd,
                       "q_dc_internal": a0["width_DC"] / 2})
    ex.to_csv(FD / "E_example_draw.csv", index=False)
    pd.DataFrame({"anchor_ref": te.ref.to_numpy()[c]}).to_csv(FD / "E_example_anchors.csv", index=False)
    pd.DataFrame(np.corrcoef((Yte - pte).T), index=TARGETS, columns=TARGETS).to_csv(FD / "E_residual_corr.csv")


# ---------------------------------------------------------------- F
def woven_overlap(Xtr, Ytr, gtr, dc_tr):
    T, X, _ = s10.design(Xtr, "woven")
    e = pd.read_csv(RESULTS / "dml_propensity_woven.csv")["propensity_woven"].to_numpy()
    keep = (e > 0.05) & (e < 0.95)
    rows = []
    outs = {t: Ytr[:, j] for j, t in enumerate(TARGETS)} | {"DC": dc_tr}
    for oc, Y in outs.items():
        th, se = s10.plr(Y[keep], T[keep], X.to_numpy()[keep], gtr[keep], True, reps=5)
        rows.append({"outcome": oc, "theta_overlap": th, "se_overlap": se, "n_overlap": int(keep.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(FD / "F_woven_overlap.csv", index=False)
    print(df.round(3))


# ---------------------------------------------------------------- G
def partial_residuals(Xtr, Ytr):
    ctrl = [c for c in Xtr.columns if c.startswith("s_") or c.startswith("f_")] + ["woven"]
    out = {}
    for x, other in [("log_mass", "log_thickness"), ("log_thickness", "log_mass")]:
        C = Xtr[ctrl + [other]].to_numpy()
        rx = Xtr[x].to_numpy() - LinearRegression().fit(C, Xtr[x]).predict(C)
        for j in [2, 3]:  # B-bias, S-warp
            ry = Ytr[:, j] - LinearRegression().fit(C, Ytr[:, j]).predict(C)
            out[f"{x}|{TARGETS[j]}|x"] = rx
            out[f"{x}|{TARGETS[j]}|y"] = ry
    pd.DataFrame(out).to_csv(FD / "G_partial_residuals.csv", index=False)


# ---------------------------------------------------------------- H
def _ale_one(Xtr, Ytr, j, b, edges):
    col = "f_elastane"
    x = Xtr[col].to_numpy()
    m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, random_state=2024, n_jobs=1)
    idx = np.random.default_rng(b).integers(0, len(Xtr), len(Xtr))
    mm = m.fit(Xtr.iloc[idx], Ytr[idx, j])
    eff = [0.0]
    for k in range(1, len(edges)):
        inb = (x > edges[k - 1]) & (x <= edges[k]) if k > 1 else (x >= edges[0]) & (x <= edges[1])
        if inb.sum() == 0:
            eff.append(eff[-1]); continue
        hi, lo = Xtr[inb].copy(), Xtr[inb].copy()
        hi[col], lo[col] = edges[k], edges[k - 1]
        hi["elastane_present"], lo["elastane_present"] = float(edges[k] > 0), float(edges[k - 1] > 0)
        eff.append(eff[-1] + np.mean(mm.predict(hi) - mm.predict(lo)))
    eff = np.array(eff)
    w = np.histogram(x, bins=edges)[0]
    eff = eff - np.sum(0.5 * (eff[1:] + eff[:-1]) * w) / w.sum()
    return [{"target": TARGETS[j], "boot": b, "x": edges[k], "ale": v} for k, v in enumerate(eff)]


def ale_elastane(Xtr, Ytr):
    x = Xtr["f_elastane"].to_numpy()
    edges = np.r_[0.0, np.unique(np.quantile(x[x > 0], np.linspace(0, 1, 11)))]
    out = Parallel(n_jobs=12)(delayed(_ale_one)(Xtr, Ytr, j, b, edges) for j in [3, 4, 5] for b in range(30))
    pd.DataFrame([r for rows in out for r in rows]).to_csv(FD / "H_ale_elastane.csv", index=False)


# ---------------------------------------------------------------- I
def silhouettes(refs):
    rows = []
    with zipfile.ZipFile(ROOT / "dataset_for_eg2024.zip") as z:
        names = [i.filename for i in z.infolist() if "/scenes/" in i.filename and i.filename.endswith(".json")]
        for r in refs:
            for src in ["GT", "MechMet", "MechIm"]:
                hit = [n for n in names if f"cusick_scenes/{src}/" in n and n.endswith(f"{r}.json")]
                if not hit:
                    continue
                b = np.asarray(json.loads(z.read(hit[0]))["projected_border_positions"])
                b = b - b.mean(0)
                o = np.argsort(np.arctan2(b[:, 1], b[:, 0]))
                for k, (u, v) in enumerate(b[o]):
                    rows.append({"ref": r, "source": src, "k": k, "x": u, "y": v})
    pd.DataFrame(rows).to_csv(FD / "I_silhouettes.csv", index=False)


def main_late() -> None:
    """Re-run only the ALE and silhouette steps (H, I)."""
    tr, _ = load_raw()
    Xtr, Ytr = engineer(tr), targets_log(tr, "_MechMet")
    print("== H ALE"); ale_elastane(Xtr, Ytr)
    print("== I silhouettes"); silhouettes(["SED-0321", "SED-0874", "SED-0988"])


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    gtr = duplicate_groups(tr)
    oof = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["oof"] for n in ENSEMBLE], axis=0)
    pte = np.mean([np.load(RESULTS / "preds" / f"preds_{n}.npz")["test"] for n in ENSEMBLE], axis=0)
    pairs = pd.read_csv(RESULTS / "drape_surrogate_oof.csv")
    sur = joblib.load(RESULTS / "drape_surrogate.joblib")
    cf = s07.crossfit_surrogates(pairs, sur["features"], sur["model"])
    mt = [cf[r] for r in te.ref]
    ld = np.log(te["mass_gsm"].to_numpy() / 1000)
    base_dc = s07.dc_of(pte, ld, mt)
    grad = np.zeros_like(pte)
    for j in range(6):
        e = np.zeros(6); e[j] = 0.05
        grad[:, j] = (s07.dc_of(pte + e, ld, mt) - s07.dc_of(pte - e, ld, mt)) / 0.1
    gt_dc = pairs[pairs.source == "GT"].set_index("ref")["drape_coefficient"].loc[te.ref].to_numpy()
    pd.DataFrame({"ref": te.ref, "dc_ens_surrogate": base_dc, "gt_dc": gt_dc}).to_csv(FD / "test_dc_ensemble.csv", index=False)
    dc_tr = sur["model"].predict(np.column_stack([Ytr, np.log(tr["mass_gsm"] / 1000)]))

    print("== A tiers"); tiers(tr, te, Xtr, Xte, Ytr, Yte)
    print("== B anchors"); anchor_policies(te, Xte, Yte, pte, base_dc, grad, gt_dc)
    print("== C Peirce"); peirce(tr, te, Xtr, Ytr, gtr)
    print("== D ROC"); roc(Xtr, Xte)
    print("== E ACDC draws"); acdc_draws(te, Yte, pte, base_dc, grad, gt_dc, Ytr, oof, tr)
    print("== F woven overlap"); woven_overlap(Xtr, Ytr, gtr, dc_tr)
    print("== G partial residuals"); partial_residuals(Xtr, Ytr)
    print("== H ALE"); ale_elastane(Xtr, Ytr)
    print("== I silhouettes"); silhouettes(["SED-0321", "SED-0874", "SED-0988"])


if __name__ == "__main__":
    import sys
    main_late() if "--late" in sys.argv else main()
