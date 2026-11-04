"""Step 1: download three PhysioNet databases and cut them into labelled 5 s single-lead clips.

    python code/data.py

Output: data/clips.npz holding X (clips x 625 samples, millivolts) and, for every clip, its database, record,
patient, start time, AHA rhythm category and shock label (1 shock, 0 no shock, -1 intermediate).
"""
import numpy as np, pandas as pd, wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from config import RAW, CLIPS, FS, WIN, DATABASES, COARSE_VF_UV, ASYSTOLE_UV, RAPID_VT_BPM

# Rhythm label in the annotation file -> rhythm group. None = drop (noise, paced or unclear rhythm).
RHYTHM = {"N": "NSR", "NSR": "NSR",
          "VF": "VF", "VFIB": "VF", "VFL": "VF", "VT": "VT", "ASYS": "ASYS",
          **{r: "OTHER" for r in ["AF", "AFIB", "AFL", "SVTA", "SBR", "B", "BI", "BII", "T", "NOD", "PREX", "IVR", "AB", "VER"]},
          **{r: None for r in ["NOISE", "P", "PM", "HGEA"]}}
SHOCK = {"coarse VF": 1, "rapid VT": 1, "NSR": 0, "other non-shockable": 0, "asystole": 0, "fine VF": -1, "other VT": -1}
SAME_PATIENT = {"mitdb/202": "mitdb/201"}  # MIT-BIH 201 and 202 come from the same person
LEAD = {"cudb": "ECG", "vfdb": "ECG", "mitdb": "MLII"}
_B, _A = butter(4, [0.5, 30], btype="band", fs=FS)


def download():
    """Fetch the databases from PhysioNet (about 130 MB) unless they are already in data/physionet."""
    for db in DATABASES:
        folder = RAW / db
        if not any(folder.glob("*.hea")):
            print(f"downloading {db} ...")
            wfdb.dl_database(db, str(folder))


def preprocess(x, fs):
    """Resample to 125 Hz and band-pass 0.5-30 Hz (removes baseline wander and muscle noise)."""
    up, down = (1, 2) if fs == 250 else (25, 72) if fs == 360 else (FS, int(fs))
    return filtfilt(_B, _A, resample_poly(np.nan_to_num(x), up, down))


def rhythm_track(db, rec, n):
    """Rhythm group of every sample, from the rhythm-change annotations (and CUDB's [ ] VF-episode markers).

    CUDB has rhythm notes in only some records, so whatever it leaves unmarked, including the stretch after a VF
    episode until the next note, stays None (dropped) instead of being assumed to be sinus rhythm.
    """
    a = wfdb.rdann(str(RAW / db / rec), "atr")
    track = np.full(n, None, dtype=object)
    changes = [(s, RHYTHM.get(x.strip("\x00")[1:], None)) for s, x in zip(a.sample, a.aux_note) if x.startswith("(")]
    for (s, r), (e, _) in zip(changes, changes[1:] + [(n, None)]):
        track[s:e] = r
    if db == "cudb":
        starts = [s for s, c in zip(a.sample, a.symbol) if c == "["]
        ends = [s for s, c in zip(a.sample, a.symbol) if c == "]"]
        for s in starts:
            e = next((x for x in ends if x > s), n)
            track[s:e] = "VF"
            nxt = next((c for c, _ in changes if c > e), n)
            track[e:nxt] = None
    return track


def amplitude_uv(x):
    """Median peak-to-peak amplitude over 1 s sub-windows, in microvolts."""
    return 1000 * np.median([np.ptp(x[i:i + FS]) for i in range(0, len(x), FS)])


def rate_bpm(x):
    """Heart rate from the autocorrelation: the earliest lag (40-300 bpm) whose peak reaches 70% of the strongest.

    Checked against MIT-BIH beat labels: median error 0.6 bpm, 99% agreement on the >= 180 bpm split.
    """
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    pk, _ = find_peaks(ac[:int(1.5 * FS) + 1])
    pk = pk[(pk >= int(0.2 * FS)) & (ac[pk] > 0)]
    return 60 * FS / pk[np.argmax(ac[pk] >= 0.7 * ac[pk].max())] if len(pk) else 0.0


def aha_category(group, x):
    """Rhythm group -> AHA category. VF is split by amplitude, VT by rate."""
    if group == "VF":
        a = amplitude_uv(x)
        return "coarse VF" if a >= COARSE_VF_UV else "fine VF" if a >= ASYSTOLE_UV else None
    if group == "VT":
        return "rapid VT" if rate_bpm(x) >= RAPID_VT_BPM else "other VT"
    return {"NSR": "NSR", "OTHER": "other non-shockable", "ASYS": "asystole"}[group]


def build_clips():
    X, meta = [], []
    for db in DATABASES:
        for hea in sorted((RAW / db).glob("*.hea")):
            rec = hea.stem
            sig, f = wfdb.rdsamp(str(RAW / db / rec))
            if LEAD[db] not in f["sig_name"]:
                continue  # MIT-BIH 102 and 104 have no MLII lead (paced records)
            raw = sig[:, f["sig_name"].index(LEAD[db])]
            track = rhythm_track(db, rec, len(raw))
            x = preprocess(raw, f["fs"])
            track = track[(np.arange(len(x)) * f["fs"] / FS).astype(int)]
            pid = SAME_PATIENT.get(f"{db}/{rec}", f"{db}/{rec}")
            for s in range(0, len(x) - WIN + 1, WIN):  # non-overlapping 5 s clips
                g = set(track[s:s + WIN])
                if len(g) != 1 or None in g:
                    continue  # the rhythm changes inside the clip, or it is noise / unlabelled
                cat = aha_category(g.pop(), x[s:s + WIN])
                if cat is None:
                    continue
                X.append(x[s:s + WIN].astype(np.float32))
                meta.append(dict(db=db, record=rec, patient=pid, start_s=s / FS, category=cat, shock=SHOCK[cat]))
    meta = pd.DataFrame(meta)
    CLIPS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIPS, X=np.stack(X), **{c: meta[c].values for c in meta})

    counts = meta.pivot_table(index="category", columns="db", values="start_s", aggfunc="count", fill_value=0)
    counts["clips"] = counts.sum(axis=1)
    counts["patients"] = meta.groupby("category")["patient"].nunique()
    print(counts.reindex(list(SHOCK)).fillna(0).astype(int).to_string())
    print(f"\n{len(meta)} clips from {meta.patient.nunique()} patients -> {CLIPS}")


def load_clips():
    """Clips as a dict of arrays; X is clipped to +-5 mV and given a channel axis, ready for Conv1D."""
    d = dict(np.load(CLIPS, allow_pickle=True))
    d["X"] = np.clip(d["X"], -5, 5)[..., None].astype(np.float32)
    return d


if __name__ == "__main__":
    download()
    build_clips()
