# Fabric reference calibration

Core analysis code for the study **How do small reference sets improve metadata-based predictions of fabric stiffness and simulated drape?**

The workflow predicts six log stiffness parameters from fabric metadata, evaluates their simulated-drape consequences, and studies calibration with small sets of released reference fabrics. The analysed collection contains 1,275 training records, 194 matched GT-scene reference fabrics and 584 released Cusick scenes.

## Data

Download the original `dataset_for_eg2024.zip` from the [source study's official project page](https://henardoel.github.io/henardoel/index_eg2024.html), then place it in the repository root. The file analysed here has SHA-256:

```text
bc65ccde56b153c45ce8c8fc39e9ec129c9a83c2a1df4841ee95fc944d8681f6
```

The dataset and its scans, scene files and rendered images remain with their original provider. This repository does not redistribute those materials, manuscripts, reference PDFs, trained model files or prediction arrays. Stage 01 reads tables and scene JSON directly from the ZIP and does not execute code supplied inside the archive.

Source study: H. Domínguez-Elvira et al., *Practical Methods to Estimate Fabric Mechanics from Metadata*, Computer Graphics Forum 43 (2024), e15029. [DOI: 10.1111/cgf.15029](https://doi.org/10.1111/cgf.15029).

## Environment

The recorded study environment used Python 3.12; its installed versions are listed in `requirements-recorded.txt`. The core-model installation recipe is `requirements.txt`. The optional interpretation recipe is `requirements-shap.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -r requirements-shap.txt
python scripts/verify_release.py
```

Activate the virtual environment using the command appropriate to your operating system before installing packages. Several stages use multiple CPU workers and may take substantial time. Reproduction involves model fitting; no model fitting is performed by the release verifier.

The recorded machine contained NumPy 1.26.4 and SHAP 0.52.0 together. SHAP 0.52.0 declares NumPy >=2 in its package metadata, so these recorded pins cannot serve as a normal clean-install recipe. The optional recipe uses SHAP 0.49.1 with NumPy 1.26.4 and Numba 0.61.2. It supports installation of the interpretation stage without changing the core-model pins. This compatibility recipe has dependency-resolution checks; the full training and SHAP calculations have not been repeated under it. `requirements-recorded.txt` documents the actual recorded installation rather than an installation recommendation.

## Reproduce the analyses

Run from the repository root, in this order. All outputs are created under `data/processed/` and `results/`.

```bash
python scripts/01_extract_data.py
python scripts/02_benchmark_20models.py
python scripts/03_improvement_probes.py
python scripts/04_image_features.py
python scripts/05_drape_surrogate.py
python scripts/06_multimodal_probe.py
python scripts/07_drape_space_eval.py
python scripts/08_shift_and_conformal.py
python scripts/09_anchor_recalibration.py
python scripts/10_dml_causal.py
python scripts/11_significance_shap.py
python scripts/12_acdc_pipeline.py
python scripts/13_extra_analyses.py
python scripts/14_si_data.py
python scripts/15_nonlinear_dc_validation.py --reps 100
python scripts/27_interval_baselines.py
python scripts/plot_calibration_summary.py
```

| Stages | Main outputs |
|---|---|
| 01 | Harmonized metadata, released parameter labels and scene-derived references |
| 02–03 | Twenty-learner benchmark and exploratory label/model variants |
| 04–06 | Scan-texture descriptors, grouped drape surrogate and multimodal variants |
| 07–09 | Fabric-excluding drape evaluation, residual intervals and correction-budget comparisons |
| 10–12 | Adjusted associations, paired bootstrap/SHAP and pooled calibration ablations |
| 13–15 | Group/family and selection diagnostics, scaling/ALE, split sensitivity and nonlinear drape checks |
| 27 | Budget-matched interval constructions (split conformal, jackknife+, Student-t and in-sample radii) on the anchor draws of stage 12 |

Stage numbers follow the study's script numbering. Numbers 16–26 are document-preparation steps that are not part of this release.

`plot_calibration_summary.py` provides a small reusable plotting example. It reads computed result tables; it does not recreate the manuscript's pictorial workflow or redistribute third-party scan/render assets.

## Evaluation definitions

- Training labels and GT-scene references retain their released identities. The GT values are simulator parameterizations read from the source scenes.
- Shared label or metadata vectors define operational dependence groups. Structure-family holdout uses a different ensemble from fabric-group validation.
- Reference calibration is retrospective within the available collection. ACDC residual intervals use leave-one-anchor-out scores with a full-anchor centre; their empirical coverage is evaluated directly. Stage 27 scores split-conformal and jackknife+ intervals on the same anchor draws for comparison.
- Material coefficients are adjusted observational associations. The drape evaluator reproduces the released simulation configuration.

The release retains the implemented analysis and random seeds. Documentation wording is aligned with these definitions; code preparation does not refit models or change numerical expressions.

## License

The study-specific code is released under the MIT license in `LICENSE`. The license covers this repository's code and documentation only. Original datasets, scans, scene files and source-study materials retain their original ownership and terms.
