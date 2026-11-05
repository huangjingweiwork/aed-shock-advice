"""v2 calibration study: how much does full-integer int8 depend on WHICH clips calibrate it?

    python code/calib_study.py --all      5 folds in parallel (needs results/keras/fold*/ from run_experiments.py)
    python code/calib_study.py --pick     choose the rule on VALIDATION patients only, then write the test results

For every trained model of every fold, the float32 model is converted to int8 with 4 calibration rules x 3 seeds:
  random300      300 random training clips (the v1 rule)
  random1000     1000 random training clips
  stratified300  60 clips from each of the 5 AHA categories, so rare rhythms (asystole) are represented
  nosat300       300 random training clips, excluding clips that hit the +-5 mV clipping limit (artefacts)
The rule is chosen by fidelity on the validation patients: the fewest int8 decisions that differ from float32.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, contextlib, io, json, subprocess, sys
import numpy as np, pandas as pd
from config import RESULTS, FOLDS

RULES = ["random300", "random1000", "stratified300", "nosat300"]
SEEDS = [0, 1, 2]
FAMS = ["small", "large", "large_p75", "large_p50", "large_p35", "scratch_k50", "scratch_k35", "large_u75"]
CATS = ["coarse VF", "rapid VT", "NSR", "other non-shockable", "asystole"]
OUT = RESULTS / "calib"


def calib_idx(rule, seed, tr, X, cat, k):
    rng = np.random.default_rng(1000 * k + seed)
    if rule == "random300":
        return rng.choice(tr, 300, replace=False)
    if rule == "random1000":
        return rng.choice(tr, 1000, replace=False)
    if rule == "stratified300":
        return np.concatenate([rng.choice(tr[cat[tr] == c], min(60, (cat[tr] == c).sum()), replace=False) for c in CATS])
    if rule == "nosat300":
        ok = tr[np.abs(X[tr]).max(axis=(1, 2)) < 4.9]
        return rng.choice(ok, 300, replace=False)
    raise ValueError(rule)


def run_fold(k):
    import keras
    from data import load_clips
    from run_experiments import split
    from compress import to_tflite_int8, run_tflite
    d = load_clips()
    X, cat = d["X"], d["category"]
    tr, va, te = split(d, k)
    out = {}
    for fam in FAMS:
        m = keras.models.load_model(RESULTS / "keras" / f"fold{k}" / f"{fam}.keras", compile=False)
        for rule in RULES:
            for seed in SEEDS:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    blob = to_tflite_int8(m, X[calib_idx(rule, seed, tr, X, cat, k)])
                out[f"{fam}__{rule}__s{seed}__val"] = run_tflite(blob, X[va]).ravel()
                out[f"{fam}__{rule}__s{seed}__test"] = run_tflite(blob, X[te]).ravel()
        print(f"fold {k}: {fam} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"fold{k}.npz", **out)


def pick():
    from data import load_clips
    from evaluate import aha_table
    d = load_clips()
    cat, pat, y = d["category"], d["patient"], d["shock"]
    rows = []
    pooled = {}  # (fam, rule, seed) -> test probabilities pooled over folds
    for k in range(FOLDS):
        raw = np.load(RESULTS / "raw" / f"fold{k}.npz")
        cs = np.load(OUT / f"fold{k}.npz")
        va, te = raw["val_idx"], raw["test_idx"]
        for fam in FAMS:
            f32v = raw[f"{fam}_fp32__val"] >= 0.5
            for rule in RULES:
                for seed in SEEDS:
                    pv = cs[f"{fam}__{rule}__s{seed}__val"]
                    rows.append(dict(fold=k, fam=fam, rule=rule, seed=seed, val_flips=int(((pv >= .5) != f32v).sum()),
                                     val_n=len(va)))
                    pooled.setdefault((fam, rule, seed), [[], []])
                    pooled[(fam, rule, seed)][0].append(te)
                    pooled[(fam, rule, seed)][1].append(cs[f"{fam}__{rule}__s{seed}__test"])
    r = pd.DataFrame(rows)
    by_rule = r.groupby("rule").val_flips.sum().sort_values()
    best = by_rule.index[0]
    print("validation decisions changed by int8, summed over 5 folds x 8 models x 3 seeds:\n" + by_rule.to_string())
    print("chosen rule:", best)
    # test results per (model, rule, seed): passed AHA rows and scores
    res = []
    for (fam, rule, seed), (idx, p) in pooled.items():
        idx, p = np.concatenate(idx), np.concatenate(p)
        t = aha_table(cat[idx], pat[idx], p >= 0.5)
        sc = dict(zip(t["category"], t["score"]))
        res.append(dict(model=fam, rule=rule, seed=seed, passed=int(t["passes"].fillna(False).astype(bool).sum()),
                        **{c: sc.get(c) for c in CATS}))
    res = pd.DataFrame(res)
    res.to_csv(RESULTS / "tables" / "calib_study_test.csv", index=False)
    r.to_csv(RESULTS / "tables" / "calib_study_val.csv", index=False)
    spread = res.groupby(["model", "rule"]).agg(passed_min=("passed", "min"), passed_max=("passed", "max"),
                                                 vf_min=("coarse VF", "min"), vf_max=("coarse VF", "max"),
                                                 asys_min=("asystole", "min"), asys_max=("asystole", "max"))
    print(spread.round(3).to_string())
    json.dump(dict(chosen_rule=best, val_flips=by_rule.to_dict()), open(RESULTS / "tables" / "calib_choice.json", "w"),
              indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pick", action="store_true")
    a = ap.parse_args()
    if a.all:
        OUT.mkdir(parents=True, exist_ok=True)
        procs = [subprocess.Popen([sys.executable, __file__, "--fold", str(k)], stdout=open(OUT / f"fold{k}.log", "w"),
                                  stderr=subprocess.STDOUT) for k in range(FOLDS)]
        print("exit codes:", [p.wait() for p in procs])
    elif a.pick:
        pick()
    else:
        run_fold(a.fold)
