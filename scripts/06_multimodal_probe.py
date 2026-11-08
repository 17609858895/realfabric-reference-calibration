"""Exploratory metadata plus scan-texture comparisons. Image descriptors are standardized and compressed inside each training fold; reference scores are retrospective."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROC, RESULTS, duplicate_groups, engineer, load_raw, metrics_block, targets_log

warnings.filterwarnings("ignore")
SEED = 2024


class FuseImage(BaseEstimator, RegressorMixin):
    """Metadata columns pass through; image columns -> impute, scale, PCA(k) fitted on the training fold."""

    def __init__(self, base=None, meta_cols=None, img_cols=None, k=20):
        self.base, self.meta_cols, self.img_cols, self.k = base, meta_cols, img_cols, k

    def _z(self, X, fit=False):
        I = X[self.img_cols].to_numpy()
        if fit:
            self.pp_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(self.k, random_state=0)).fit(I)
        Z = self.pp_.transform(I) if self.k else np.empty((len(X), 0))
        return np.column_stack([X[self.meta_cols].to_numpy(), Z])

    def fit(self, X, Y):
        self.m_ = clone(self.base).fit(self._z(X, fit=True), Y)
        return self

    def predict(self, X):
        return self.m_.predict(self._z(X))


def run(est, Xtr, Ytr, Xte, Yte, folds):
    oof = np.zeros_like(Ytr)
    for a, b in folds:
        oof[b] = clone(est).fit(Xtr.iloc[a], Ytr[a]).predict(Xtr.iloc[b])
    pte = clone(est).fit(Xtr, Ytr).predict(Xte)
    return oof, pte


def main() -> None:
    tr, te = load_raw()
    im = pd.read_csv(PROC / "image_features.csv").drop(columns="split").set_index("ref")
    img_cols = list(im.columns)
    Xtr = pd.concat([engineer(tr), im.loc[tr.ref].reset_index(drop=True)], axis=1)
    Xte = pd.concat([engineer(te), im.loc[te.ref].reset_index(drop=True)], axis=1)
    meta_cols = list(engineer(tr).columns)
    Ytr, Yte = targets_log(tr, "_MechMet"), targets_log(te)
    folds = list(GroupKFold(5).split(Xtr, Ytr, duplicate_groups(tr)))
    lgbp = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=4, random_state=SEED)
    bases = {
        "LightGBM": MultiOutputRegressor(LGBMRegressor(**lgbp), n_jobs=6),
        "CatBoost": CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, loss_function="MultiRMSE",
                                      verbose=0, random_seed=SEED, thread_count=16),
        "KRR": make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02)),
    }
    rows, store = [], {}
    for bn, base in bases.items():
        for k in [0, 10, 20, 40]:
            if bn == "KRR" and k:
                b2 = make_pipeline(StandardScaler(), KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02 * 34 / (34 + k)))
            else:
                b2 = base
            est = FuseImage(b2, meta_cols, img_cols, k)
            oof, pte = run(est, Xtr, Ytr, Xte, Yte, folds)
            store[(bn, k)] = (oof, pte)
            cv, tm = metrics_block(Ytr, oof), metrics_block(Yte, pte)
            rows.append({"base": bn, "img_pcs": k, "cv_R2log": cv["R2log_mean"], "test_R2log": tm["R2log_mean"],
                         "test_NMAE": tm["NMAE_mean"], "test_MdAPE": tm["MdAPE_mean"],
                         **{f"test_{c}": v for c, v in tm.items() if c.startswith("R2log_k")}})
            print(rows[-1]["base"], k, f"cv {rows[-1]['cv_R2log']:.3f} test {rows[-1]['test_R2log']:.3f} "
                                       f"NMAE {rows[-1]['test_NMAE']:.3f} MdAPE {rows[-1]['test_MdAPE']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "probe_multimodal.csv", index=False)
    for k in [0, 20]:
        avg = np.mean([store[(b, k)][1] for b in bases], axis=0)
        print(f"avg of 3 bases, img_pcs={k}: test R2log {metrics_block(Yte, avg)['R2log_mean']:.3f}")
    np.savez(RESULTS / "preds" / "multimodal_store.npz",
             **{f"{b}_{k}_{w}": store[(b, k)][i] for (b, k) in store for i, w in enumerate(["oof", "test"])})


if __name__ == "__main__":
    main()
