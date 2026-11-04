# Shrinking a shock-advice network to fit an AED chip

ESP3201A final project (NUS, 2026). An automated external defibrillator (AED) decides "shock" or "no shock" from 5 seconds of ECG, on a small microcontroller. We train a 1D-CNN shock-advice model, compress it with **structured pruning** and **int8 quantisation** (TensorFlow Lite), and check whether it still meets the **American Heart Association (AHA) targets for each rhythm**, while fitting an **STM32L432 (256 KiB flash, 64 KiB RAM)**.

Research questions:
1. How far can the model be compressed and still meet the AHA rhythm targets on an AED-class chip?
2. Is it better to compress a big model or to train a small one from the start?

**Key results:** see [`results/RESULTS.md`](results/RESULTS.md) and the report, [`report/report.docx`](report/report.docx).

| Model (int8 unless noted) | Flash | RAM | Fits 256/64 KiB? | AHA rows passed (of 5) |
|---|---|---|---|---|
| Large 1D ResNet, float32 (reference) | 1,975 KiB | 117 KiB | No | 3 |
| Large, int8 | 538 KiB | 29 KiB | No | 3 |
| **Pruned to 35% of filters, int8** | **93 KiB** | **10 KiB** | **Yes** | **4** |
| **35%-width model trained from scratch, int8** | **93 KiB** | **10 KiB** | **Yes** | **4** |
| Small CNN (4.6k weights), int8 | 13 KiB | 5 KiB | Yes | 3 |

- int8 quantisation: ~4x smaller and 1.3% of decisions changed. Structured pruning plus fine-tuning brings the model under the chip's budget. Unstructured pruning (zeroed weights) does not shrink the on-chip model at all.
- The 35%-width model meets every AHA row except asystole, 21x smaller than the float32 reference. Training that shape from scratch did as well as pruning.
- No model passes asystole: 113 of its 147 clips come from one patient. Overall metrics hide per-rhythm damage: the 50%-pruned model's AUROC barely moved under int8 (0.993 to 0.986), but its asystole specificity fell from 60% to 18%.

**Run it in Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huangjingweiwork/aed-shock-advice/blob/main/colab/aed_shock_advice.ipynb) — `Runtime → Run all`. The notebook writes the same scripts as `code/`, downloads the data and runs every step; a dropdown picks `quick` (fold 0, 2 epochs, checks the pipeline), `fold0` or `full`.

## Repository layout

| Folder | What is in it |
|---|---|
| `code/` | The whole pipeline, one script per step (below) |
| `results/` | Tables (`tables/*.csv`), figures (`figures/*.png`), the fold-0 `.tflite` models (`models/`), raw per-fold predictions (`raw/`) and `RESULTS.md` |
| `report/` | The final report (`report.docx`), its Markdown source and the script that builds it |
| `colab/` | `aed_shock_advice.ipynb`, the whole pipeline as one Colab notebook |

| Script | Step |
|---|---|
| `code/config.py` | All settings: paths, clip format, training budget, chip budget, AHA targets |
| `code/data.py` | Downloads CUDB, VFDB and MIT-BIH from PhysioNet; cuts 5 s single-lead clips at 125 Hz; labels them by AHA category |
| `code/models.py` | The small CNN and the large 1D ResNet |
| `code/compression.py` | Structured (L1 filter) pruning, unstructured pruning, TFLite float32 / dynamic-range / full-integer int8 conversion, memory and latency measurement |
| `code/evaluate.py` | AHA scoring with patient-bootstrap lower confidence limits; threshold selection on validation patients |
| `code/run_experiments.py` | Trains, prunes, converts and scores every model on each patient-wise fold |
| `code/summarise.py` | Pools the folds into the result tables and `RESULTS.md` |
| `code/make_figures.py` | Draws every figure in the report |
| `code/demo.py` | Runs the int8 model on ECG clips and prints "SHOCK ADVISED" / "NO SHOCK" |

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
python code/data.py                       # 1. download (~130 MB) and build data/clips.npz  (a few minutes)
python code/run_experiments.py --all      # 2. 5 folds in parallel                         (~1 h on 16+ cores)
python code/summarise.py                  # 3. tables + results/RESULTS.md
python code/make_figures.py               # 4. figures
python code/demo.py                       # 5. demo on held-out patients
```

- To check that everything runs before the full experiment: `python code/run_experiments.py --quick`, then `python code/summarise.py --quick` and `python code/make_figures.py --quick` (2 epochs, writes to `results_quick/`, about 10 min).
- One fold at a time: `python code/run_experiments.py --fold 0` (repeat for 0-4).
- Demo on any PhysioNet record: `python code/demo.py --record cudb/cu12 --start 260`. Choose a different model with `--model results/models/small_int8.tflite`.

The results in `results/` are committed, so steps 3-5 work without retraining. Training is seeded, but TensorFlow on different CPUs can give slightly different numbers.

## Method in short

- **Data.** PhysioNet CUDB, VFDB and MIT-BIH Arrhythmia; one lead, 125 Hz, band-pass 0.5-30 Hz, non-overlapping 5 s clips labelled into the AHA categories (coarse VF ≥ 200 µV, rapid VT ≥ 180 bpm; the rate cut-off is our assumption).
- **Evaluation.** Patient-wise 5-fold cross-validation (no patient is in both training and test). Each AHA row passes only if the score beats the AHA goal *and* its one-sided 90% lower confidence limit (bootstrap over patients) reaches the AHA minimum (Kerber et al., *Circulation* 1997). We report two decision thresholds: 0.5, and one chosen on validation patients.
- **Models.** Large 1D ResNet (501,537 weights) and small 3-layer CNN (4,625 weights).
- **Compression.** Structured L1-norm filter pruning (keep 75/50/35%, then fine-tune), the same narrow shapes trained from scratch, unstructured 75% magnitude pruning, and TFLite post-training quantisation (dynamic-range and full-integer int8, 300 calibration clips).
- **Chip fit.** Flash = `.tflite` file size; RAM = peak live activation memory (lower bound). On-chip latency is *estimated* from MAC count (no board was used).

## Limitations

- Asystole comes from only 6 patients, so its lower confidence limit is wide.
- No out-of-hospital AED recordings with CPR artefacts are publicly available.
- Latency on the microcontroller is estimated, not measured.

## Data and licences

ECG data: PhysioNet (Goldberger et al., 2000), CUDB, VFDB and MIT-BIH Arrhythmia Database, (open access; see each database's PhysioNet page for its licence). They are downloaded by `code/data.py` and not stored in this repository.
