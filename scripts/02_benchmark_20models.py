"""Core metadata benchmark: operational-group cross-validation and retrospective GT-scene reference scores. The twenty fixed learner configurations predict six log stiffness parameters from released training labels."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from common import RESULTS, TARGETS, duplicate_groups, engineer, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
SEED = 2024
PRED_DIR = RESULTS / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def mo(est):
    return MultiOutputRegressor(est, n_jobs=6)


def scaled(est):
    return make_pipeline(StandardScaler(), est)


def models() -> dict:
    gpk = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(0.1)
    return {
        "Ridge": scaled(Ridge(alpha=3.0)),
        "Lasso": scaled(mo(Lasso(alpha=0.005, max_iter=20000))),
        "ElasticNet": scaled(mo(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000))),
        "BayesianRidge": scaled(mo(BayesianRidge())),
        "PLS": scaled(PLSRegression(n_components=12)),
        "KNN": scaled(KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "SVR": scaled(mo(SVR(C=3.0, epsilon=0.05, gamma="scale"))),
        "KernelRidge": scaled(KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02)),
        "GPR": scaled(mo(GaussianProcessRegressor(kernel=gpk, normalize_y=True, random_state=SEED))),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=8, random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                              n_jobs=16, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.6,
                                          n_jobs=16, random_state=SEED),
        "GBRT": mo(GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                             min_samples_leaf=5, random_state=SEED)),
        "HistGB": mo(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                                                   min_samples_leaf=10, l2_regularization=1.0,
                                                   random_state=SEED)),
        "AdaBoost": mo(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=5),
                                         n_estimators=300, learning_rate=0.05, random_state=SEED)),
        "Bagging": BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=300,
                                    max_samples=0.8, n_jobs=16, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                                colsample_bytree=0.7, min_child_weight=3, reg_lambda=2.0, n_jobs=16,
                                multi_strategy="one_output_per_tree", random_state=SEED),
        "LightGBM": mo(LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                     verbose=-1, n_jobs=4, random_state=SEED)),
        "CatBoost": CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, loss_function="MultiRMSE",
                                      l2_leaf_reg=3.0, verbose=0, random_seed=SEED, thread_count=16),
        "MLP": scaled(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                   max_iter=2000, early_stopping=True, validation_fraction=0.15,
                                   random_state=SEED)),
    }


def fit_predict(est, Xtr, Ytr, Xte):
    m = clone(est)
    m.fit(Xtr, Ytr)
    p = m.predict(Xte)
    return np.asarray(p).reshape(len(Xte), -1)


def main() -> None:
    tr, te = load_raw()
    Xtr, Xte = engineer(tr), engineer(te)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    groups = duplicate_groups(tr)
    print(f"train {Xtr.shape}, test {Xte.shape}, duplicate-aware groups: {len(np.unique(groups))}")
    folds = list(GroupKFold(n_splits=5).split(Xtr, Ytr, groups))
    np.save(RESULTS / "cv_groups.npy", groups)

    rows = []
    for name, est in models().items():
        t0 = time.time()
        oof = np.zeros_like(Ytr)
        for k, (a, b) in enumerate(folds):
            oof[b] = fit_predict(est, Xtr.iloc[a], Ytr[a], Xtr.iloc[b])
        ptr = fit_predict(est, Xtr, Ytr, Xtr)
        pte = fit_predict(est, Xtr, Ytr, Xte)
        np.savez(PRED_DIR / f"preds_{name}.npz", oof=oof, train=ptr, test=pte)
        cv = metrics_block(Ytr, oof)
        fold_r2 = [metrics_block(Ytr[b], oof[b])["R2log_mean"] for _, b in folds]
        trm = metrics_block(Ytr, ptr)
        tm = metrics_block(Yte, pte)
        rows.append({"model": name, "train_R2log": trm["R2log_mean"], "cv_R2log": cv["R2log_mean"],
                     "cv_R2log_min_fold": min(fold_r2), "test_R2log": tm["R2log_mean"],
                     "gap_train_test": trm["R2log_mean"] - tm["R2log_mean"],
                     "test_RMSElog": tm["RMSElog_mean"], "test_MdAPE": tm["MdAPE_mean"],
                     "test_NMAE": tm["NMAE_mean"], "cv_NMAE": cv["NMAE_mean"],
                     **{f"test_{k}": v for k, v in tm.items() if k.startswith("R2log_k")},
                     "sec": time.time() - t0})
        r = rows[-1]
        print(f"{name:14s} trainR2 {r['train_R2log']:.3f}  cvR2 {r['cv_R2log']:.3f}  testR2 {r['test_R2log']:.3f}"
              f"  testNMAE {r['test_NMAE']:.3f}  MdAPE {r['test_MdAPE']:.3f}  ({r['sec']:.0f}s)")

    # published EG2024 predictions on the same test fabrics
    pub = pd.read_csv(__import__("common").PROC / "test_paper_preds.csv").set_index("ref").loc[te["ref"]]
    sp = pd.read_csv(__import__("common").PROC / "scene_params.csv")
    for meth in ["MechMet", "MechIm"]:
        P = pub[[f"{t}_pred{meth}" for t in TARGETS]].to_numpy(dtype=float)
        P = np.log(np.clip(P, 1.0, None))  # published values contain a few negative stiffnesses
        tm = metrics_block(Yte, P)
        rows.append({"model": f"EG2024-{meth} (csv)", "test_R2log": tm["R2log_mean"], "test_RMSElog": tm["RMSElog_mean"],
                     "test_MdAPE": tm["MdAPE_mean"], "test_NMAE": tm["NMAE_mean"],
                     **{f"test_{k}": v for k, v in tm.items() if k.startswith("R2log_k")}})
        sc = sp[sp.source == meth].set_index("ref")[TARGETS].copy()
        sc[TARGETS[:3]] *= 1e8
        ok = te["ref"].isin(sc.index).to_numpy()
        P2 = np.log(sc.loc[te["ref"][ok]].to_numpy())
        tm2 = metrics_block(Yte[ok], P2)
        rows.append({"model": f"EG2024-{meth} (scene)", "test_R2log": tm2["R2log_mean"],
                     "test_RMSElog": tm2["RMSElog_mean"], "test_MdAPE": tm2["MdAPE_mean"],
                     "test_NMAE": tm2["NMAE_mean"],
                     **{f"test_{k}": v for k, v in tm2.items() if k.startswith("R2log_k")}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "benchmark_20models.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 3):
        print(df[["model", "train_R2log", "cv_R2log", "test_R2log", "gap_train_test", "test_NMAE", "test_MdAPE"]]
              .sort_values("test_R2log", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
