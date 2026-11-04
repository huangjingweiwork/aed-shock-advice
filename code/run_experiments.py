"""Step 2: train, compress and score every model, with patient-wise 5-fold cross-validation.

    python code/run_experiments.py --all         all 5 folds as parallel processes (about 30-40 min, 16+ cores)
    python code/run_experiments.py --fold 0      one fold only
    python code/run_experiments.py --quick       fold 0 with 2 epochs, to check the pipeline runs (results_quick/)

Per fold:
  1. train the small model and the large model (float32)
  2. structured pruning of the large model (keep 75/50/35% of filters), then fine-tune
  3. the same narrow shapes trained from scratch (does pruning beat simply starting small?)
  4. unstructured pruning of the large model (75% of weights set to zero), then fine-tune
  5. convert every model to .tflite: float32, dynamic-range int8 (small/large only) and full-integer int8
  6. score every .tflite file on the validation patients (to pick a threshold later) and the test patients
Writes results/raw/fold{k}.npz (predictions) and, for fold 0, the .tflite files and their sizes.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
import keras
from sklearn.model_selection import StratifiedGroupKFold
from config import (RESULTS, SEED, FOLDS, EPOCHS, FT_EPOCHS, BATCH, N_CALIB, KEEPS, SCRATCH_KEEPS, SPARSITY)
from data import load_clips
from models import small, large, count_macs
from compression import (prune_structured, prune_unstructured, KeepPruned, sparsity_of, to_tflite_fp32,
                         to_tflite_dynamic, to_tflite_int8, run_tflite, peak_activation_bytes, gzip_bytes)


def split(d, k):
    """Fold k: test patients, plus training patients split again into train / validation (1 in 6 patients).

    Whole patients go to one side only, so the model is always tested on people it has never seen.
    """
    y = d["shock"] == 1
    has_shock = pd.Series(y).groupby(d["patient"]).transform("max").values  # balance shockable patients
    folds = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(d["X"], has_shock, d["patient"])
    tr, te = list(folds)[k]
    tr = tr[d["shock"][tr] >= 0]  # intermediate rhythms (fine VF, slow VT) are scored but never trained on
    pats = np.unique(d["patient"][tr])
    val_p = np.random.default_rng(SEED + k).choice(pats, max(1, len(pats) // 6), replace=False)
    in_val = np.isin(d["patient"][tr], val_p)
    return tr[~in_val], tr[in_val], te


def fit(model, X, y, tr, va, epochs, lr, callbacks=()):
    """Adam + binary cross-entropy, classes weighted equally, early stopping on validation loss."""
    w = {0: len(tr) / (2 * (y[tr] == 0).sum()), 1: len(tr) / (2 * (y[tr] == 1).sum())}
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy")
    stop = keras.callbacks.EarlyStopping("val_loss", patience=6 if lr >= 1e-3 else 4, restore_best_weights=True)
    model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs, batch_size=BATCH, class_weight=w,
              verbose=0, callbacks=[stop, *callbacks])
    return model


def run_fold(k, quick, out):
    t0 = time.time()
    d = load_clips()
    X, y = d["X"], (d["shock"] == 1).astype(np.float32)
    tr, va, te = split(d, k)
    epochs, ft_epochs = (2, 1) if quick else (EPOCHS, FT_EPOCHS)
    keras.utils.set_random_seed(SEED + k)
    calib = X[np.random.default_rng(SEED + k).choice(tr, N_CALIB, replace=False)]
    log = lambda msg: print(f"[fold {k} {(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)
    log(f"train {len(tr)}  val {len(va)}  test {len(te)} clips")

    probs = {}  # variant name -> (validation probabilities, test probabilities)
    keras_prob = lambda m: tuple(m.predict(X[idx], batch_size=512, verbose=0).ravel() for idx in (va, te))
    nets = {"small": fit(small(), X, y, tr, va, epochs, 1e-3)}
    log("small trained")
    nets["large"] = fit(large(), X, y, tr, va, epochs, 1e-3)
    log("large trained")
    for keep in KEEPS:
        tag = f"large_p{int(keep * 100)}"
        p = prune_structured(nets["large"], keep)
        probs[f"{tag}_noft_fp32"] = keras_prob(p)  # straight after cutting, before any retraining
        nets[tag] = fit(p, X, y, tr, va, ft_epochs, 3e-4)
        log(f"{tag} pruned + fine-tuned")
    for keep in SCRATCH_KEEPS:
        nets[f"scratch_k{int(keep * 100)}"] = fit(large(keep=keep), X, y, tr, va, epochs, 1e-3)
        log(f"scratch_k{int(keep * 100)} trained")
    tag = f"large_u{int(SPARSITY * 100)}"
    u, masks = prune_unstructured(nets["large"], SPARSITY)
    probs[f"{tag}_noft_fp32"] = keras_prob(u)
    nets[tag] = fit(u, X, y, tr, va, ft_epochs, 3e-4, [KeepPruned(masks)])
    log(f"{tag} pruned + fine-tuned (sparsity {sparsity_of(nets[tag]):.3f})")

    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for fam, m in nets.items():
        blobs = {"fp32": to_tflite_fp32(m), "int8": to_tflite_int8(m, calib)}
        if fam in ("small", "large"):
            blobs["dyn"] = to_tflite_dynamic(m)
        for kind, blob in blobs.items():
            name = f"{fam}_{kind}"
            probs[name] = (run_tflite(blob, X[va]), run_tflite(blob, X[te]))
            if k == 0:
                (mdir / f"{name}.tflite").write_bytes(blob)
                sizes[name] = dict(params=m.count_params(), macs=count_macs(m), sparsity=round(sparsity_of(m), 4),
                                   file_kib=len(blob) / 1024, gzip_kib=gzip_bytes(blob) / 1024,
                                   peak_ram_kib=peak_activation_bytes(blob) / 1024)
        if k == 0 and fam == "large":
            m.save(mdir / "large_fp32.keras")  # kept for the weight / filter figures
    log("converted and scored all .tflite files")

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw / f"fold{k}.npz", val_idx=va, test_idx=te,
                        **{f"{n}__val": v for n, (v, _) in probs.items()},
                        **{f"{n}__test": t for n, (_, t) in probs.items()})
    if k == 0:
        json.dump(sizes, open(out / "sizes.json", "w"), indent=1)
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads per process (0 = TensorFlow default)")
    a = ap.parse_args()
    out = RESULTS.parent / "results_quick" if a.quick else RESULTS
    if a.all:  # one process per fold, CPU threads shared out between them
        threads = a.threads or max(1, (os.cpu_count() or 4) // FOLDS)
        (out / "raw").mkdir(parents=True, exist_ok=True)
        procs = [subprocess.Popen([sys.executable, __file__, "--fold", str(k), "--threads", str(threads)],
                                  stdout=open(out / "raw" / f"fold{k}.log", "w"), stderr=subprocess.STDOUT)
                 for k in range(FOLDS)]
        codes = [p.wait() for p in procs]
        print("exit codes per fold:", codes)
        sys.exit(max(codes))
    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    run_fold(0 if a.fold is None else a.fold, a.quick, out)


if __name__ == "__main__":
    main()
