"""Clip-length study: does the model decide better from a longer stretch of ECG?

    python code/window_study.py --build        cut clips of every length from the five databases (data/windows/)
    python code/window_study.py --all          train and score every length x fold x seed, several at a time
    python code/window_study.py --summarise    results/windows/summary.csv, aha.csv and the figure

The main study uses 5 s clips. Here the same pipeline is repeated for clips of 3 to 10 s, with everything else
held fixed:
  - the same recordings, labels and filters; clips are cut back to back, as in the main study
  - the same 5 minutes of each rhythm per person (the cap becomes 300 s / clip length clips)
  - the same people in each test and validation group as the main 5 s study, so the lengths are compared on
    exactly the same people
  - the two chip-sized designs: the small CNN and the 50%-width network trained from scratch (the best model
    that fits), each as a float32 and a full-integer int8 .tflite file, calibrated with the main study's rule
  - four training seeds per fold, because pass/fail often turns on a point or less
Both designs end in global average pooling, so the same layers (and the same weights count) read any length.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, contextlib, io, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from config import DATA, RESULTS, FS, FOLDS, SEED, EPOCHS, CAP, WIN_S, MCU_HZ, MACS_PER_CYCLE

LENGTHS = (3, 4, 5, 6, 7, 8, 9, 10)  # seconds
SEEDS = (0, 1, 2, 3)
CAP_S = CAP * WIN_S                   # 300 s: at most 5 minutes of each rhythm per person
WDIR = DATA / "windows"
OUT = RESULTS / "windows"


def clips_path(w):
    return WDIR / f"clips_{w}s.npz"


def build(w):
    from data import collect, cap_per_person
    X, meta = collect(FS * w)
    keep = cap_per_person(meta, CAP_S // w)
    X, meta = X[keep], meta.loc[keep].reset_index(drop=True)
    WDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(clips_path(w), X=X, **{c: meta[c].values for c in meta})
    print(f"{w} s: {len(meta)} clips from {meta.patient.nunique()} people", flush=True)


def load(w):
    d = dict(np.load(clips_path(w), allow_pickle=True))
    d["X"] = np.clip(d["X"], -5, 5)[..., None].astype(np.float32)
    return d


def groups():
    """Test and validation people of every fold, taken from the main 5 s study so all lengths share them."""
    from data import load_clips
    from run_experiments import split
    d = load_clips()
    out = []
    for k in range(FOLDS):
        tr, va, te = split(d, k)
        out.append((set(d["patient"][te]), set(d["patient"][va])))
    return out


def run_job(w, k, seed):
    import keras, tensorflow as tf
    from models import small, large, count_macs
    from run_experiments import fit
    from calib_study import calib_idx
    from compress import to_tflite_fp32, to_tflite_int8, run_tflite, peak_activation_bytes
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    d = load(w)
    X, y, pat = d["X"], (d["shock"] == 1).astype(np.float32), d["patient"]
    test_p, val_p = groups()[k]
    in_te, in_va = np.isin(pat, list(test_p)), np.isin(pat, list(val_p))
    te = np.where(in_te)[0]
    va = np.where(in_va & (d["shock"] >= 0))[0]
    tr = np.where(~in_te & ~in_va & (d["shock"] >= 0))[0]
    calib = X[calib_idx("nosat300", 0, tr, X, d["category"], k)]
    n = X.shape[1]
    out, sizes = {"test_idx": te}, {}
    for fam, build_model in [("small", lambda: small(n)), ("scratch_k50", lambda: large(n, keep=0.5))]:
        keras.utils.set_random_seed(SEED + k + 100 * seed)
        m = fit(build_model(), X, y, tr, va, EPOCHS, 1e-3)
        for kind, blob in [("fp32", to_tflite_fp32(m)), ("int8", to_tflite_int8(m, calib))]:
            out[f"{fam}_{kind}"] = run_tflite(blob, X[te]).ravel()
            sizes[f"{fam}_{kind}"] = dict(params=m.count_params(), macs=count_macs(m), file_kib=len(blob) / 1024,
                                          peak_ram_kib=peak_activation_bytes(blob) / 1024)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"w{w}_s{seed}_fold{k}.npz", **out)
    json.dump(sizes, open(OUT / f"w{w}_s{seed}_fold{k}_sizes.json", "w"), indent=1)


def run_all(parallel):
    jobs = [(w, k, s) for s in SEEDS for w in LENGTHS for k in range(FOLDS)
            if not (OUT / f"w{w}_s{s}_fold{k}.npz").exists()]
    print(f"{len(jobs)} jobs, {parallel} at a time", flush=True)
    t0 = time.time()

    def one(job):
        w, k, s = job
        log = OUT / "logs" / f"w{w}_s{s}_fold{k}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        code = subprocess.call([sys.executable, __file__, "--job", str(w), str(k), str(s)],
                               stdout=open(log, "w"), stderr=subprocess.STDOUT)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {w} s, seed {s}, fold {k}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(parallel) as ex:
        codes = list(ex.map(one, jobs))
    sys.exit(max(codes, default=0))


def summarise():
    from sklearn.metrics import roc_auc_score
    from evaluate import aha_table
    rows, aha = [], []
    for w in LENGTHS:
        d = load(w)
        cat, pat, shock = d["category"], d["patient"], d["shock"]
        for s in SEEDS:
            files = [OUT / f"w{w}_s{s}_fold{k}.npz" for k in range(FOLDS)]
            if not all(f.exists() for f in files):
                continue
            z = [np.load(f) for f in files]
            idx = np.concatenate([q["test_idx"] for q in z])
            sizes = json.load(open(OUT / f"w{w}_s{s}_fold0_sizes.json"))
            for model in ("small_fp32", "small_int8", "scratch_k50_fp32", "scratch_k50_int8"):
                p = np.concatenate([q[model] for q in z])
                shocked = p >= 0.5
                t = aha_table(cat[idx], pat[idx], shocked)
                t.insert(0, "seed", s), t.insert(0, "model", model), t.insert(0, "length_s", w)
                aha.append(t)
                known = shock[idx] >= 0
                sz = sizes[model]
                rows.append(dict(length_s=w, seed=s, model=model, clips=int(len(idx)), people=len(np.unique(pat[idx])),
                                 auroc=roc_auc_score(shock[idx][known], p[known]),
                                 sens=shocked[known & (shock[idx] == 1)].mean(),
                                 spec=(~shocked[known & (shock[idx] == 0)]).mean(),
                                 passed=int(t.passes.fillna(False).astype(bool).sum()),
                                 file_kib=sz["file_kib"], peak_ram_kib=sz["peak_ram_kib"], macs=sz["macs"],
                                 mcu_ms_low=sz["macs"] / (MACS_PER_CYCLE[1] * MCU_HZ) * 1000,
                                 mcu_ms_high=sz["macs"] / (MACS_PER_CYCLE[0] * MCU_HZ) * 1000))
    summary, aha = pd.DataFrame(rows), pd.concat(aha, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    aha.to_csv(OUT / "aha.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    wide = aha[aha.goal.notna()].pivot_table(index=["model", "length_s"], columns="category", values="score",
                                             aggfunc="mean")
    print("\nmean score over seeds\n", (100 * wide).round(1).to_string())


def figure():
    """Fig. 10: score per AHA row against clip length, for the two chip-sized int8 models (mean of the seeds, with
    each seed as a faint dot), plus the number of rows passed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_figures as mf  # same style and colours as the other report figures
    from config import AHA
    aha = pd.read_csv(OUT / "aha.csv")
    summary = pd.read_csv(OUT / "summary.csv")
    models = [("scratch_k50_int8", "50%-width from scratch + int8", mf.C2, "o"),
              ("small_int8", "Small + int8", mf.C4, "s")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    for ax, (cat, metric, goal, _) in zip(axes.flat, AHA):
        for m, label, col, mk in models:
            t = aha[(aha.model == m) & (aha.category == cat)]
            mean = t.groupby("length_s").score.mean()
            ax.scatter(t.length_s, 100 * t.score, color=col, s=9, alpha=0.35, lw=0)
            ax.plot(mean.index, 100 * mean.values, color=col, lw=2, marker=mk, ms=4, label=label)
        ax.axhline(100 * goal, color=mf.INK, lw=1.2, ls="--")
        ax.set_title(f"{cat} (goal {100 * goal:.0f}%)", loc="left", fontsize=8.5)
        ax.axvline(5, color=mf.MUTED, lw=0.8, ls=":")
    ax = axes.flat[5]
    for m, label, col, mk in models:
        t = summary[summary.model == m]
        mean = t.groupby("length_s").passed.mean()
        ax.scatter(t.length_s, t.passed, color=col, s=9, alpha=0.35, lw=0)
        ax.plot(mean.index, mean.values, color=col, lw=2, marker=mk, ms=4)
    ax.set_ylim(-0.3, 5.3)
    ax.set_title("AHA rows passed (of 5)", loc="left", fontsize=8.5)
    ax.axvline(5, color=mf.MUTED, lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("clip length (s)")
        ax.set_xticks(LENGTHS)
    for ax in axes[:, 0]:
        ax.set_ylabel("score (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=mf.INK, lw=1.2, ls="--"))
    labels.append("AHA goal")
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    mf.save(fig, "fig10_clip_length.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-one", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", type=int, default=7)
    ap.add_argument("--job", type=int, nargs=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if a.figure:
        return figure()
    if a.build_one:
        build(a.build_one)
    elif a.build:  # one process per length, four at a time (each holds every uncapped clip in memory)
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda w: subprocess.call([sys.executable, __file__, "--build-one", str(w)]), LENGTHS))
    elif a.job:
        run_job(*a.job)
    elif a.all:
        run_all(a.parallel)
    elif a.summarise:
        summarise()


if __name__ == "__main__":
    main()
