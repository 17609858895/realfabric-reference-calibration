"""Step 01 - extract and harmonise the EG2024 RealFabric dataset.

Reads the original ZIP without unpacking images or running any packaged code.
Outputs (data/processed/):
  train_clean.csv      1,275 training fabrics, harmonised schema
  test_clean.csv       test fabrics with metadata mapped to the training schema
  scene_params.csv     simulator parameters stored in every Cusick scene JSON (GT / MechIm / MechMet)
  drape_metrics.csv    drape-shape descriptors computed from projected border positions
  test_paper_preds.csv the published MechIm / MechMet predictions for the test fabrics
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "dataset_for_eg2024.zip"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["kBendingWarp", "kBendingWeft", "kBendingBias", "kStretchWarp", "kStretchWeft", "kStretchBias"]
STRUCTS = ["Brushed Back Fleece", "Crepe", "Dobby", "Fleece", "French Terry", "Interlock", "Jacquard", "Jersey",
           "Mesh", "Milano", "Ottoman", "Pile", "Pique", "Plain", "Rib", "Satin", "Terry Cloth", "Tricot", "Twill",
           "Velour"]
FIBRES = ["Wool", "Others", "Polyester", "Angora", "Cashmere", "Ramie", "Mohair", "Nylon (polyamide)", "Rayon",
          "metallic", "Polyurethane (Spandex)", "Bamboo", "Vegetal Fibre", "Acetate", "Alpaca", "Cotton", "Silk",
          "Acrylic", "hemp", "modal", "Flax"]
# test-file fibre names -> training schema
FIBRE_MAP = {
    "nylon": "Nylon (polyamide)", "Polyamide": "Nylon (polyamide)", "elastane": "Polyurethane (Spandex)",
    "viscose rayon": "Rayon", "Tencel": "Rayon", "organic cotton": "Cotton", "cotton": "Cotton",
    "polyester": "Polyester", "Polyester DTY": "Polyester", "CD Polyester": "Polyester", "CD": "Polyester",
    "Lycra T400": "Polyester", "silk": "Silk", "wool": "Wool", "virgin wool": "Wool", "cashmereWool": "Cashmere",
    "flax": "Flax", "metallic": "metallic", "other fibers": "Others", "acrylic": "Acrylic", "triacetate": "Acetate",
    "modal": "modal", "hemp": "hemp",
}


def load_train(z: zipfile.ZipFile) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(z.read("dataset_for_eg2024/dataset_train.csv")))
    df = df.rename(columns={"density": "mass_gsm"})
    df["structure_class"] = np.where(df["woven"] == 1, "woven", "knit")
    df["structure_name"] = df[STRUCTS].idxmax(axis=1).where(df[STRUCTS].sum(axis=1) > 0, "Unknown")
    df["brand"] = df["ref"].str.split("-").str[0]
    return df


def load_test(z: zipfile.ZipFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(io.BytesIO(z.read("dataset_for_eg2024/dataset_test.csv")), header=[0, 1])
    raw = raw.iloc[1:].reset_index(drop=True)  # first row holds the 'internal_id' label
    ref = raw[("Unnamed: 0_level_0", "Unnamed: 0_level_1")].astype(str)
    md = raw["metadata"]
    out = pd.DataFrame({"ref": ref})
    for s in STRUCTS:
        out[s] = (md["structure_name"] == s).astype(int)
    out["basicKnit"] = (md["structure"] == "basicKnit").astype(int)
    out["woven"] = (md["structure"] == "woven").astype(int)
    out["thickness"] = md["thickness"].astype(float)
    out["mass_gsm"] = md["mass"].astype(float)
    fib = pd.DataFrame(0.0, index=out.index, columns=FIBRES)
    for col in [c for c in md.columns if c.startswith("pct-")]:
        name = col[4:]
        fib[FIBRE_MAP[name]] += md[col].astype(float) / 100.0
    out = pd.concat([out, fib], axis=1)
    out["structure_class"] = np.where(out["woven"] == 1, "woven", "knit")
    out["structure_name"] = md["structure_name"].values
    out["family"] = md["family"].values
    out["brand"] = out["ref"].str.split("-").str[0]
    preds = pd.DataFrame({"ref": ref})
    for meth in ["MechIm", "MechMet"]:
        for t in TARGETS:
            preds[f"{t}_pred{meth}"] = raw[(f"pred_{meth}", t)].astype(float)
    return out, preds


def drape_descriptors(border: np.ndarray, radius_disk: float = 0.09, radius_fabric: float = 0.15) -> dict:
    """Cusick-style descriptors of a projected drape silhouette (positions in metres)."""
    xy = border - border.mean(axis=0, keepdims=True)
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(ang)
    xy, ang = xy[order], ang[order]
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    a_disk = np.pi * radius_disk ** 2
    a_fab = np.pi * radius_fabric ** 2
    r = np.hypot(x, y)
    # number of nodes = number of local maxima of the smoothed radius profile
    grid = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    rr = np.interp(grid, ang, r, period=2 * np.pi)
    k = np.ones(9) / 9
    rs = np.convolve(np.r_[rr[-4:], rr, rr[:4]], k, mode="valid")
    peaks = int(np.sum((rs > np.roll(rs, 1)) & (rs > np.roll(rs, -1)) & (rs > rs.mean())))
    return {
        "drape_area": area,
        "drape_coefficient": (area - a_disk) / (a_fab - a_disk),
        "r_mean": r.mean(), "r_std": r.std(), "r_min": r.min(), "r_max": r.max(),
        "n_nodes": peaks,
    }


def load_scenes(z: zipfile.ZipFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    prows, drows = [], []
    # file stems look like ..._GT_SED-1050 or ..._<long model tag>_SED-1050; keep the trailing fabric ID only
    pat = re.compile(r"cusick_scenes/(GT|MechIm|MechMet)/scenes/mech_params_test_to_simulate_.*?([A-Za-z0-9]+-\d+)\.json$")
    for info in z.infolist():
        m = pat.search(info.filename)
        if not m:
            continue
        src, ref = m.group(1), m.group(2)
        d = json.loads(z.read(info.filename))
        p = d["params"]
        prows.append({"ref": ref, "source": src, **{k: p[k] for k in p}})
        border = np.asarray(d["projected_border_positions"], dtype=float)
        drows.append({"ref": ref, "source": src, "n_border": len(border), **drape_descriptors(border)})
    return pd.DataFrame(prows), pd.DataFrame(drows)


def main() -> None:
    with zipfile.ZipFile(ZIP) as z:
        tr = load_train(z)
        te, preds = load_test(z)
        params, drape = load_scenes(z)
    tr.to_csv(OUT / "train_clean.csv", index=False)
    te.to_csv(OUT / "test_clean.csv", index=False)
    preds.to_csv(OUT / "test_paper_preds.csv", index=False)
    params.to_csv(OUT / "scene_params.csv", index=False)
    drape.to_csv(OUT / "drape_metrics.csv", index=False)
    print("train", tr.shape, "test", te.shape, "scene params", params.shape, "drape", drape.shape)
    print(params.groupby("source").size())


if __name__ == "__main__":
    main()
