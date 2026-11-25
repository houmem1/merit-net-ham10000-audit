# MERIT-Net under a leakage-audited, lesion-grouped protocol (HAM10000)

Splits, configurations, per-fold predictions, code and analysis outputs for
the article

> H. Slimi, A. Ben Aissa, M. Sayadi. *MERIT-Net under a leakage-audited,
> lesion-grouped protocol: a paired evaluation of multi-scale attention
> pooling and class-balanced evidential learning for skin-lesion
> classification on HAM10000.* BMC Medical Imaging (under review).

Every number, table and figure of the article is generated from the files in
this repository by `code/rev_analysis.py` and `code/rev_make_assets.py`.

## Contents

| Folder | Content |
|---|---|
| `splits/split_assignments.csv` | One row per HAM10000 image: `image_id`, `lesion_id`, `dx`, `y`, `dx_type` (reference standard), `partition` (`test`, `cv1`..`cv5`). This is the frozen lesion-grouped protocol (also Additional file 2 of the article). |
| `splits/kaggle_split.json` | The same split as index lists (`te_idx`, `folds_va`), as consumed by the training code. Indices refer to the images sorted by `image_id`. |
| `splits/split_imagelevel.json`, `split_imageclean.json` | Image-level control split (leaky arm) and its clean arm (pool images sharing a lesion with the test set removed). |
| `configs/configs.json` | Configuration of every run (23 matched-budget, 7 performance-oriented, 7 revision experiments) with the label used in the article. |
| `predictions/derm_<tag>.json` | For every run: configuration, test labels, per-fold validation / test / PH2 class probabilities and uncertainties, per-fold and ensemble metrics, training curves. `derm-<tag>.log` is the training log. |
| `analysis/rev_analysis.json` | Output of the statistical analysis (fold-matched ensembles, lesion-cluster bootstrap CIs, paired contrasts, McNemar tests, Holm adjustment, recalibration controls, referral counts, melanoma-vs-nevus audit, PH2, duplicate sensitivity analysis, leakage control). |
| `analysis/dupscreen.json` | Cross-partition perceptual-hash screen (pHash and dHash, eight dihedral transforms). |
| `analysis/abhishek2025_*.csv` | The 18 same-lesion pairs with different lesion identifiers published by Abhishek et al. (Sci Data 2025), and their location in our partitions. |
| `code/` | Training code (`kernel_body.py`, run with `local_run.py` or pushed to Kaggle with `push_run.py`), campaign orchestration, and the analysis / asset scripts. |

## Run tags

* Matched budget (23): `main` = MERIT-Net (MSAP encoder + CB-EDL), `merit-ce`
  = MSAP encoder + CE, `cnx-ce`, `cnx-cbedl` (2x2 factorial, 5 folds);
  `r50-ce`, `dense121-ce`, `effb3-ce`, `deit3s-ce`, `swinv2t-ce`; `cnx-wce`,
  `cnx-focal`, `cnx-cbce`, `cnx-edl`; ablations `m-gap`, `m-attplain`,
  `m-single`, `m-fusemean`, `m-lam0`, `m-lam01`, `m-beta9999`, `m-edl`,
  `m-effb3`, `m-224` (3 folds).
* Performance-oriented, exploratory (7, 5 folds): `boost` = P1, `boost2` =
  P2, `boost3` = P3, `boost4` = P4, `boost5` = P5, `screen1` = B1,
  `screen2` = B2.
* Revision experiments (7, 3 folds): `ctl-imgsplit`, `ctl-imgclean`,
  `m-edl-lam01`, `cnx-cbedl-lam01`, `main-s43`, `m-lam01-s43`,
  `merit-ce-s43`.

## Reproducing the analysis

```bash
pip install numpy pandas scipy scikit-learn matplotlib
# place predictions/derm_<tag>.json under results/<tag>/ and the analysis
# inputs under results/revision/ (paths are defined at the top of the scripts)
python code/rev_analysis.py      # -> rev_analysis.json
python code/rev_make_assets.py   # -> tables, figures, numbers
```

## Reproducing the training

The images are **not** redistributed here. Obtain HAM10000 from
https://doi.org/10.7910/DVN/DBW86T and PH2 from
https://www.fc.up.pt/addi/ph2%20database.html, build the resized caches with
`code/local_prep.py`, then run e.g.

```bash
python code/local_run.py main '{"folds_run": 5, "split_file": "splits/kaggle_split.json"}'
```

Requirements: Python 3.11, PyTorch 2.x, timm 1.0, scikit-learn, OpenCV.

## Licences

Code: MIT (see `LICENSE`). Splits, predictions and analysis outputs: CC BY
4.0. HAM10000 and PH2 remain under the licences of their providers; the
split file contains only the public image and lesion identifiers and
diagnoses of the HAM10000 metadata.
