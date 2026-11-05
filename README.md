# Shrinking a shock-advice network to fit an AED chip

ESP3201A final project (NUS, 2026). An automated external defibrillator (AED) decides "shock" or "no shock" from 5 seconds of ECG, on a small microcontroller. We train a 1D-CNN shock-advice model, compress it with **structured pruning** and **int8 quantisation** (TensorFlow Lite), and check whether it still meets the **American Heart Association (AHA) targets for each rhythm**, while fitting an **STM32L432 (256 KiB flash, 64 KiB RAM)**.

Research questions:
1. How far can the model be compressed and still meet the AHA rhythm targets on an AED-class chip?
2. Is it better to compress a big model or to train a small one from the start?

**Key results:** see [`results/RESULTS.md`](results/RESULTS.md). The written report is submitted separately.

## Data: five PhysioNet databases, 863 people

| | Clips | People |
|---|---|---|
| Shockable (coarse VF, rapid VT) | 1,712 | 79 |
| Non-shockable (normal rhythm, other rhythms, asystole) | 7,529 | 732 |
| In-between (fine VF, slower VT; reported, no AHA goal) | 508 | 107 |
| **Total** | **9,749** | **863** |

- Sources: CUDB, VFDB and MIT-BIH Arrhythmia (rhythm-annotated recordings), the Sudden Cardiac Death Holter Database (VF recorded at the moment of arrest) and the PhysioNet/CinC Challenge 2015 ICU alarms (including 447 false alarms: noise that fooled a bedside monitor into calling VF).
- At most 60 clips (5 minutes) per person per rhythm, so no one long recording dominates.
- Asystole comes from 28 people; per-source counts are in [`data/dataset_table.csv`](data/dataset_table.csv).

## Results (5-fold patient-wise cross-validation, threshold 0.5)

| Model (int8 unless noted) | Flash | RAM | Fits 256/64 KiB? | AHA rows passed (of 5) |
|---|---|---|---|---|
| Large 1D ResNet, float32 (reference) | 1,975 KiB | 117 KiB | No | 2 |
| Large, int8 | 538 KiB | 29 KiB | No | 1 |
| Pruned to 50% of filters, int8 | 160 KiB | 15 KiB | Yes | 2 |
| **50%-width model trained from scratch, int8** | **160 KiB** | **15 KiB** | **Yes** | **3** |
| Pruned to 35% of filters, int8 | 93 KiB | 10 KiB | Yes | 2 |
| 35%-width model trained from scratch, int8 | 93 KiB | 10 KiB | Yes | 1 |
| Small CNN (4.6k weights), int8 | 13 KiB | 5 KiB | Yes | 2 |

- **Best model that fits the chip:** the 50%-width network trained from scratch, in int8. It is 12× smaller than the reference and passes fast VT, normal rhythm and other non-shockable. It misses coarse VF by 1.4 points (88.6% vs 90%) and asystole by 1 point (94.0% vs 95%).
- **The bottleneck is int8, not size.** The same model in float32 passes 4 of 5. Full-integer int8 changes 2–7% of the compressed models' decisions, mostly costing VF sensitivity. Quantisation-aware training is the next step.
- **Calibration matters.** int8 results depended strongly on which clips calibrate the activations. Of 4 rules compared on validation patients, calibrating on 300 clips without clipping artefacts changed the fewest decisions (30% fewer than random clips) and is used for every int8 result ([`code/calib_study.py`](code/calib_study.py)).
- **Train small beats prune:** at the same shape, the from-scratch models matched or beat the pruned ones, and lost fewer decisions to int8 (194 vs 297 at 50% width).
- **Unstructured pruning** (zeroed weights) does not shrink the on-chip model at all (still 538 KiB).
- **The new data is harder.** On the first version of this project (100 patients from 3 databases, commit `cc7bc5d`) the reference passed 3 of 5. VF at the moment of arrest (SDDB) is caught 82–89% of the time, against 92–97% for the original databases.

**Run it in Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huangjingweiwork/aed-shock-advice/blob/main/colab/aed_shock_advice.ipynb) — `Runtime → Run all`. The notebook writes the same scripts as `code/`, downloads the data and runs every step; a dropdown picks `quick` (fold 0, 2 epochs, checks the pipeline), `fold0` or `full`.

## Repository layout

| Folder | What is in it |
|---|---|
| `code/` | The whole pipeline, one script per step (below) |
| `dl_extra.py`, `check_sddb_vf.py` | Download the two added databases; plot the SDDB VF onsets to check the labels by eye |
| `data/` | The dataset table and the two label-check images (the ECG data itself is downloaded, not stored) |
| `results/` | Tables (`tables/*.csv`), figures (`figures/*.png`), the fold-0 `.tflite` models (`models/`), raw per-fold predictions (`raw/`), the calibration study (`calib/`) and `RESULTS.md` |
| `colab/` | `aed_shock_advice.ipynb`, the whole pipeline as one Colab notebook |

| Script | Step |
|---|---|
| `code/config.py` | All settings: paths, clip format, per-person cap, training budget, chip budget, AHA targets |
| `code/data.py` | Downloads CUDB, VFDB and MIT-BIH; cuts 5 s single-lead clips at 125 Hz from all five databases; labels them by AHA category; applies the per-person cap |
| `code/models.py` | The small CNN and the large 1D ResNet |
| `code/compress.py` | Structured (L1 filter) pruning, unstructured pruning, TFLite float32 / dynamic-range / full-integer int8 conversion, memory and latency measurement |
| `code/evaluate.py` | AHA scoring with patient-bootstrap lower confidence limits; threshold selection on validation patients |
| `code/run_experiments.py` | Trains, prunes, converts and scores every model on each patient-wise fold |
| `code/calib_study.py` | Compares 4 int8 calibration rules × 3 seeds and picks one on validation patients |
| `code/summarise.py` | Pools the folds into the result tables and `RESULTS.md` |
| `code/make_figures.py` | Draws every figure in the report |
| `code/demo.py` | Runs an int8 model on ECG clips and prints "SHOCK ADVISED" / "NO SHOCK" |

## Setup

Python 3.12 on Windows, macOS or Linux; CPU only (no GPU needed).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

All commands are run from the repository root.

```bash
python dl_extra.py                        # 1. download CinC 2015 and SDDB (~2 GB)
python code/data.py                       # 2. download CUDB, VFDB, MIT-BIH and build data/clips.npz
python code/run_experiments.py --all      # 3. 5 folds in parallel                         (~1-2 h on 16+ cores)
python code/calib_study.py --all          # 4. optional: the calibration comparison
python code/calib_study.py --pick
python code/summarise.py                  # 5. tables + results/RESULTS.md
python code/make_figures.py               # 6. figures
python code/demo.py                       # 7. demo on held-out patients
```

- To check that everything runs before the full experiment: `python code/run_experiments.py --quick`, then `python code/summarise.py --quick` and `python code/make_figures.py --quick` (2 epochs, writes to `results_quick/`).
- One fold at a time: `python code/run_experiments.py --fold 0` (repeat for 0-4).
- Demo on any PhysioNet record: `python code/demo.py --record cudb/cu12 --start 260`. Choose a different model with `--model results/models/small_int8.tflite`.

The results in `results/` are committed, so steps 5-7 work without retraining (the demo needs the data from steps 1-2). Training is seeded, but TensorFlow on different CPUs can give slightly different numbers.

## Method in short

- **Data.** One lead, 125 Hz, band-pass 0.5-30 Hz, non-overlapping 5 s clips labelled into the AHA categories (coarse VF ≥ 200 µV, rapid VT ≥ 180 bpm; the rate cut-off is our assumption). SDDB VF is taken from the published onset, for a length checked by eye per record; SDDB normal and AF clips come from at least 10 minutes before onset. CinC 2015 clips are the 5 s before each expert-reviewed alarm.
- **Evaluation.** Patient-wise 5-fold cross-validation (no person is in both training and test). Each AHA row passes only if the score beats the AHA goal *and* its one-sided 90% lower confidence limit (bootstrap over patients) reaches the AHA minimum (Kerber et al., *Circulation* 1997). We report two decision thresholds: 0.5, and one chosen on validation patients.
- **Models.** Large 1D ResNet (501,537 weights; BatchNorm momentum 0.9) and small 3-layer CNN (4,625 weights).
- **Compression.** Structured L1-norm filter pruning (keep 75/50/35%, then fine-tune), the same narrow shapes trained from scratch, unstructured 75% magnitude pruning, and TFLite post-training quantisation (dynamic-range and full-integer int8, calibrated on 300 artefact-free training clips).
- **Chip fit.** Flash = `.tflite` file size; RAM = peak live activation memory (lower bound). On-chip latency is *estimated* from MAC count (no board was used).

## Limitations

- None of the recordings come from AED pads; no out-of-hospital AED recordings with CPR artefacts are publicly available.
- Rapid VT comes from only 16 people, almost all from one database (VFDB).
- Each CinC 2015 record is treated as a different person (the challenge publishes no patient IDs).
- Latency on the microcontroller is estimated, not measured.

## Data and licences

ECG data: PhysioNet (Goldberger et al., 2000): CUDB, VFDB, MIT-BIH Arrhythmia, the Sudden Cardiac Death Holter Database and the PhysioNet/Computing in Cardiology Challenge 2015 (open access; see each database's PhysioNet page for its licence). They are downloaded by `dl_extra.py` and `code/data.py` and not stored in this repository.
