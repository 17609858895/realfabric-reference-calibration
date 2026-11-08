"""Shared data loading, feature engineering, grouping and metrics for the RealFabric study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TARGETS = ["kBendingWarp", "kBendingWeft", "kBendingBias", "kStretchWarp", "kStretchWeft", "kStretchBias"]
SHORT = {"kBendingWarp": "B-warp", "kBendingWeft": "B-weft", "kBendingBias": "B-bias",
         "kStretchWarp": "S-warp", "kStretchWeft": "S-weft", "kStretchBias": "S-bias"}
STRUCTS = ["Brushed Back Fleece", "Crepe", "Dobby", "Fleece", "French Terry", "Interlock", "Jacquard", "Jersey",
           "Mesh", "Milano", "Ottoman", "Pile", "Pique", "Plain", "Rib", "Satin", "Terry Cloth", "Tricot", "Twill",
           "Velour"]
FIBRES = ["Wool", "Others", "Polyester", "Angora", "Cashmere", "Ramie", "Mohair", "Nylon (polyamide)", "Rayon",
          "metallic", "Polyurethane (Spandex)", "Bamboo", "Vegetal Fibre", "Acetate", "Alpaca", "Cotton", "Silk",
          "Acrylic", "hemp", "modal", "Flax"]
# fibre families used to pool rare fibres (keeps compositional information, reduces sparsity)
FIBRE_FAMILY = {
    "cellulose_natural": ["Cotton", "Flax", "hemp", "Ramie", "Vegetal Fibre", "Bamboo"],
    "cellulose_regen": ["Rayon", "modal", "Acetate"],
    "protein": ["Wool", "Cashmere", "Angora", "Mohair", "Alpaca", "Silk"],
    "polyester": ["Polyester"],
    "polyamide": ["Nylon (polyamide)"],
    "elastane": ["Polyurethane (Spandex)"],
    "other_synth": ["Acrylic", "metallic", "Others"],
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(PROC / "train_clean.csv")
    te = pd.read_csv(PROC / "test_clean.csv")
    sp = pd.read_csv(PROC / "scene_params.csv")
    gt = sp[sp.source == "GT"].set_index("ref")[TARGETS].copy()
    gt[TARGETS[:3]] *= 1e8  # bending stored in N*m in scenes; CSV convention is x1e8
    te = te.merge(gt, left_on="ref", right_index=True, how="inner")
    return tr, te


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Metadata-only descriptors (no image or target information)."""
    X = pd.DataFrame(index=df.index)
    X["woven"] = df["woven"].astype(float)
    for s in STRUCTS:
        X[f"s_{s}"] = df[s].astype(float)
    t_mm = df["thickness"] * 1e3
    X["log_thickness"] = np.log(t_mm)
    X["log_mass"] = np.log(df["mass_gsm"])
    X["log_bulk_density"] = np.log(df["mass_gsm"] / t_mm)  # g m^-2 mm^-1 ~ packing/compactness
    for fam, cols in FIBRE_FAMILY.items():
        X[f"f_{fam}"] = df[cols].sum(axis=1)
    p = df[FIBRES].clip(lower=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    X["fibre_entropy"] = ent
    X["n_fibres"] = (p > 0).sum(axis=1)
    X["elastane_present"] = (df["Polyurethane (Spandex)"] > 0).astype(float)
    return X


def targets_log(df: pd.DataFrame, suffix: str = "") -> np.ndarray:
    return np.log(df[[t + suffix for t in TARGETS]].to_numpy(dtype=float))


def duplicate_groups(tr: pd.DataFrame) -> np.ndarray:
    """Connected components of fabrics that share a bending triplet, a stretch triplet, or identical metadata.

    These operational dependence groups keep shared vectors together; physical specimen identity is not established by matching alone.
    """
    n = len(tr)
    keys = [
        tr[[t + "_MechMet" for t in TARGETS[:3]]].round(6).astype(str).agg("|".join, axis=1),
        tr[[t + "_MechMet" for t in TARGETS[3:]]].round(6).astype(str).agg("|".join, axis=1),
        engineer(tr).round(6).astype(str).agg("|".join, axis=1),
    ]
    rows, cols = [], []
    for k in keys:
        first = {}
        for i, v in enumerate(k):
            if v in first:
                rows.append(i)
                cols.append(first[v])
            else:
                first[v] = i
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return lab


def metrics_block(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict:
    """Per-target and averaged metrics on log scale and on raw scale."""
    out = {}
    yt, yp = np.exp(y_log_true), np.exp(y_log_pred)
    for j, t in enumerate(TARGETS):
        a, b = y_log_true[:, j], y_log_pred[:, j]
        ss = ((a - a.mean()) ** 2).sum()
        out[f"R2log_{t}"] = 1 - ((a - b) ** 2).sum() / ss
        out[f"RMSElog_{t}"] = np.sqrt(((a - b) ** 2).mean())
        out[f"MdAPE_{t}"] = np.median(np.abs(yp[:, j] - yt[:, j]) / yt[:, j])
        out[f"NMAE_{t}"] = np.mean(np.abs(yp[:, j] - yt[:, j])) / np.mean(yt[:, j])
        out[f"R2raw_{t}"] = 1 - ((yt[:, j] - yp[:, j]) ** 2).sum() / ((yt[:, j] - yt[:, j].mean()) ** 2).sum()
    for m in ["R2log", "RMSElog", "MdAPE", "NMAE", "R2raw"]:
        out[f"{m}_mean"] = np.mean([out[f"{m}_{t}"] for t in TARGETS])
    return out
