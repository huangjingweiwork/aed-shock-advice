"""Demo: give the int8 model a 5 s ECG and get the AED's answer, "SHOCK ADVISED" or "NO SHOCK".

    python code/demo.py                                  one clip of every rhythm, from patients the model never saw
    python code/demo.py --record cudb/cu12 --start 260   any PhysioNet record, from a chosen second
    python code/demo.py --model results/models/small_int8.tflite

The model is the fold-0 .tflite file, the same bytes that would be flashed to the microcontroller, and the
threshold is 0.5, as in the report's headline results (change it with --threshold).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, wfdb
from config import RESULTS, RAW, WIN, FS
from compress import run_tflite
from data import load_clips, preprocess, LEAD

DEFAULT = RESULTS / "models" / "scratch_k50_int8.tflite"  # v2: best int8 model that fits the chip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT)
    ap.add_argument("--record", help="database/record, e.g. cudb/cu12")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    blob = a.model.read_bytes()
    name = a.model.stem
    thr = a.threshold
    print(f"model {name}: {len(blob) / 1024:.0f} KiB, shock if P(shock) >= {thr:.2f}\n")

    if a.record:
        db, rec = a.record.split("/")
        sig, f = wfdb.rdsamp(str(RAW / db / rec))
        x = preprocess(sig[:, f["sig_name"].index(LEAD[db])], f["fs"])
        s = int(a.start * FS)
        clip = np.clip(x[s:s + WIN], -5, 5).astype(np.float32)[None, :, None]
        t0 = time.perf_counter()
        p = run_tflite(blob, clip)[0]
        print(f"{a.record} from {a.start:.0f} s: P(shock) = {p:.3f} -> {'SHOCK ADVISED' if p >= thr else 'NO SHOCK'}"
              f"   ({1000 * (time.perf_counter() - t0):.0f} ms on this PC)")
        return

    d = load_clips()
    raw = a.model.resolve().parent.parent / "raw" / "fold0.npz"  # the results folder this model came from
    test = np.load(raw)["test_idx"]  # patients held out of fold 0's training
    rng = np.random.default_rng()
    print(f"{'rhythm':<22}{'patient':<13}{'should':<10}{'P(shock)':>9}  answer")
    for cat in ["coarse VF", "rapid VT", "NSR", "other non-shockable", "asystole", "fine VF", "other VT"]:
        idx = test[d["category"][test] == cat]
        if not len(idx):
            continue
        i = rng.choice(idx)
        p = run_tflite(blob, d["X"][i:i + 1])[0]
        should = {1: "shock", 0: "no shock", -1: "(either)"}[int(d["shock"][i])]
        answer = "SHOCK ADVISED" if p >= thr else "NO SHOCK"
        ok = "" if d["shock"][i] < 0 else ("  ok" if (p >= thr) == (d["shock"][i] == 1) else "  WRONG")
        print(f"{cat:<22}{d['patient'][i]:<13}{should:<10}{p:>9.3f}  {answer}{ok}")


if __name__ == "__main__":
    main()
