"""Step 04 - handcrafted, interpretable texture descriptors from front/back scans (no pretrained weights).

Per image (256x256 RGB, 11 cm field):
  * FFT power spectrum: radial band energies, warp(vertical)/weft(horizontal)/diagonal sector energies,
    dominant spatial frequency along each axis (~ yarn pitch), spectral anisotropy, spectral entropy
  * GLCM (distances 1,2,4,8; 4 angles): contrast, homogeneity, energy, correlation (mean + anisotropy)
  * uniform LBP histograms (P=8,R=1; P=16,R=2)
  * colour (HSV mean/std), grey-level std, gradient-orientation coherence
Front, back and |front-back| difference descriptors are written to data/processed/image_features.csv.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from common import PROC, ROOT

ZIP = ROOT / "dataset_for_eg2024.zip"


def fft_feats(g: np.ndarray) -> dict:
    g = g - g.mean()
    w = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(g * w)) ** 2)
    n = g.shape[0]
    c = n // 2
    yy, xx = np.mgrid[-c:n - c, -c:n - c]
    r = np.hypot(xx, yy)
    th = np.mod(np.degrees(np.arctan2(yy, xx)), 180)
    F[c, c] = 0
    tot = F[r > 1].sum() + 1e-12
    out = {}
    bands = [(2, 6), (6, 12), (12, 24), (24, 48), (48, 96), (96, 128)]
    for i, (a, b) in enumerate(bands):
        out[f"fft_band{i}"] = F[(r >= a) & (r < b)].sum() / tot
    sectors = {"horiz": (th < 15) | (th >= 165), "vert": (th >= 75) & (th < 105),
               "diag45": (th >= 30) & (th < 60), "diag135": (th >= 120) & (th < 150)}
    for k, m in sectors.items():
        out[f"fft_sec_{k}"] = F[m & (r > 2)].sum() / tot
    out["fft_aniso_hv"] = np.log((out["fft_sec_horiz"] + 1e-9) / (out["fft_sec_vert"] + 1e-9))
    # dominant frequency along axes (cycles per image) ~ yarn pitch
    row = F[c, c + 2:].copy() + F[c, :c - 1][::-1][: n - c - 2]
    col = F[c + 2:, c].copy() + F[:c - 1, c][::-1][: n - c - 2]
    out["fft_peak_x"] = float(np.argmax(row) + 2)
    out["fft_peak_y"] = float(np.argmax(col) + 2)
    out["fft_peak_ratio_x"] = row.max() / (row.mean() + 1e-12)
    out["fft_peak_ratio_y"] = col.max() / (col.mean() + 1e-12)
    p = F[r > 1] / tot
    out["fft_entropy"] = float(-(p * np.log(p + 1e-20)).sum())
    out["fft_centroid"] = float((F * r).sum() / (F.sum() + 1e-12))
    return out


def glcm_feats(g8: np.ndarray) -> dict:
    q = (g8 // 8).astype(np.uint8)  # 32 levels
    d = [1, 2, 4, 8]
    a = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(q, distances=d, angles=a, levels=32, symmetric=True, normed=True)
    out = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        v = graycoprops(P, prop)  # (d, a)
        for i, di in enumerate(d):
            out[f"glcm_{prop}_d{di}"] = v[i].mean()
        out[f"glcm_{prop}_aniso"] = float(np.log((v[:, 0].mean() + 1e-9) / (v[:, 2].mean() + 1e-9)))
    return out


def lbp_feats(g8: np.ndarray) -> dict:
    out = {}
    for P, R in [(8, 1), (16, 2)]:
        L = local_binary_pattern(g8, P, R, method="uniform")
        h, _ = np.histogram(L, bins=np.arange(P + 3), density=True)
        for i, v in enumerate(h):
            out[f"lbp{P}_{i}"] = v
    return out


def colour_grad(rgb: np.ndarray, g: np.ndarray) -> dict:
    hsv = rgb2hsv(rgb)
    out = {f"hsv{c}_mean": hsv[..., i].mean() for i, c in enumerate("hsv")}
    out.update({f"hsv{c}_std": hsv[..., i].std() for i, c in enumerate("hsv")})
    out["grey_std"] = g.std()
    gy, gx = np.gradient(g)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    out["grad_coherence"] = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)
    out["grad_energy"] = jxx + jyy
    out["grad_xy_ratio"] = np.log((jxx + 1e-12) / (jyy + 1e-12))
    return out


def one_image(png_bytes: bytes) -> dict:
    rgb = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=float) / 255.0
    g = rgb @ np.array([0.299, 0.587, 0.114])
    g8 = np.clip(g * 255, 0, 255).astype(np.uint8)
    f = {}
    f.update(fft_feats(g))
    f.update(glcm_feats(g8))
    f.update(lbp_feats(g8))
    f.update(colour_grad(rgb, g))
    return f


def job(args):
    ref, split, fb, bb = args
    ff, bf = one_image(fb), one_image(bb)
    row = {"ref": ref, "split": split}
    row.update({f"F_{k}": v for k, v in ff.items()})
    row.update({f"B_{k}": v for k, v in bf.items()})
    row.update({f"D_{k}": abs(ff[k] - bf[k]) for k in ff})
    return row


def main() -> None:
    tr = pd.read_csv(PROC / "train_clean.csv")
    te = pd.read_csv(PROC / "test_clean.csv")
    tasks = []
    with zipfile.ZipFile(ZIP) as z:
        names = {i.filename.split("/")[-1]: i.filename for i in z.infolist() if i.filename.endswith(".png")
                 and "input_imgs" in i.filename}
        for split, df in [("train", tr), ("test", te)]:
            for ref in df["ref"]:
                fn, bn = names.get(f"{ref}-FRONT.png"), names.get(f"{ref}-BACK.png")
                if fn and bn:
                    tasks.append((ref, split, z.read(fn), z.read(bn)))
    print("images pairs:", len(tasks))
    with ProcessPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(job, tasks, chunksize=8))
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "image_features.csv", index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
