"""Step 1: download three PhysioNet databases and cut them into labelled 5 s single-lead clips.

    python code/data.py

Output: data/clips.npz holding X (clips x 625 samples, millivolts) and, for every clip, its database, record,
patient, start time, AHA rhythm category and shock label (1 shock, 0 no shock, -1 intermediate).
"""
import numpy as np, pandas as pd, wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from config import RAW, CLIPS, FS, WIN, DATABASES, CAP, COARSE_VF_UV, ASYSTOLE_UV, RAPID_VT_BPM

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


### v2 sources ###########################################################################################

# SDDB: VF onset (elapsed h:m:s) from the database's PhysioNet page; records 40, 42 and 49 have no VF.
SDDB_VF_ONSET = {"30": "07:54:33", "31": "13:42:24", "32": "16:45:18", "33": "04:46:19", "34": "06:35:44",
                 "35": "24:34:56", "36": "18:59:01", "37": "01:31:13", "38": "08:01:54", "39": "04:37:51",
                 "41": "02:59:24", "43": "15:37:11", "44": "19:38:45", "45": "18:09:17", "46": "03:41:47",
                 "47": "06:13:01", "48": "02:29:40", "50": "11:45:43", "51": "22:58:23", "52": "02:32:40"}
# Seconds of VF used after each onset, read by eye from data/sddb_vf_check.png (made by check_sddb_vf.py):
# 120 s where VF lasts the whole window, shorter where it turns into an organised rhythm, 0 where unclear
# (43: regular spikes over the VF; 51: organised rhythm within seconds).
SDDB_VF_S = {"30": 120, "31": 120, "32": 120, "33": 15, "34": 120, "35": 115, "36": 20, "37": 100, "38": 75,
             "39": 105, "41": 45, "43": 0, "44": 120, "45": 120, "46": 120, "47": 120, "48": 120, "50": 45,
             "51": 0, "52": 120}
SDDB_GAP_S = 600       # non-shockable clips must end at least this long before the VF onset
VENTRICULAR = set("VFE!")  # beat symbols that make a clip ineligible as plain NSR / AF
CINC_GROUP = {"Ventricular_Flutter_Fib": "VF", "Asystole": "ASYS", "Ventricular_Tachycardia": "VT_ALARM",
              "Bradycardia": "OTHER", "Tachycardia": "OTHER"}
CINC_LEADS = ("II", "I", "III", "aVF", "V", "MCL", "aVR", "aVL")


def _seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return 3600 * h + 60 * m + s


def cinc_clips():
    """PhysioNet/CinC 2015 ICU alarms. Each record ends with an expert-reviewed alarm at 300 s; the event starts
    in the 10 s before it, so the clip is the 5 s just before the alarm (295-300 s).

    True VF / asystole alarms give VF / asystole clips; true bradycardia and tachycardia alarms give other
    non-shockable clips. True VT alarms are kept as report-only "other VT": the alarm fires on 5 ventricular beats,
    which need not be the sustained VT an AED should shock. False alarms of every type are non-shockable clips
    (subtype "false alarm"): noise or other rhythms that fooled a bedside monitor.
    """
    folder = RAW / "challenge-2015_training"
    for hea in sorted(folder.glob("*.hea")):
        rec = hea.stem
        sig, f = wfdb.rdsamp(str(folder / rec))
        kind, verdict = [c.strip() for c in f["comments"][:2]]
        lead = next((l for l in CINC_LEADS if l in f["sig_name"]), None)
        if lead is None:
            continue
        raw = sig[:, f["sig_name"].index(lead)]
        w = raw[int(295 * f["fs"]):int(300 * f["fs"])]
        if len(w) < 5 * f["fs"] or np.isnan(w).any():
            continue  # signal dropout in the alarm window
        x = preprocess(raw[:int(300 * f["fs"])], f["fs"])[-WIN:]
        true = verdict.lower().startswith("true")
        group = CINC_GROUP[kind] if true else "OTHER"
        if group == "VT_ALARM":
            cat = "other VT"
        else:
            cat = aha_category(group, x)
        if cat is None:
            continue
        yield x, dict(db="cinc2015", record=rec, patient=f"cinc2015/{rec}", start_s=295.0, category=cat,
                      shock=SHOCK[cat], subtype=("true " if true else "false ") + kind)


def sddb_clips():
    """Sudden Cardiac Death Holter Database: VF from the published onset time (the per-record SDDB_VF_S seconds), and
    NSR / AF clips from the annotated rhythm well before the onset, skipping clips with ventricular beats."""
    folder = RAW / "sddb"
    for hea in sorted(folder.glob("*.hea")):
        rec = hea.stem
        sig, f = wfdb.rdsamp(str(folder / rec), channels=[0])
        x = preprocess(sig[:, 0], f["fs"])
        n = len(x)
        onset = _seconds(SDDB_VF_ONSET[rec]) if rec in SDDB_VF_ONSET else None
        pid = f"sddb/{rec}"
        if onset is not None:
            for s in range(onset * FS, min(n, (onset + SDDB_VF_S[rec]) * FS) - WIN + 1, WIN):
                cat = aha_category("VF", x[s:s + WIN])
                if cat:
                    yield x[s:s + WIN], dict(db="sddb", record=rec, patient=pid, start_s=s / FS, category=cat,
                                             shock=SHOCK[cat], subtype="VF after onset")
        a = wfdb.rdann(str(folder / rec), "ari")
        notes = [(int(s * FS / f["fs"]), x_[1:].strip("\x00")) for s, x_ in zip(a.sample, a.aux_note)
                 if x_.startswith("(") and x_[1:].strip("\x00") in ("N", "AFIB")]
        beats = np.array([int(s * FS / f["fs"]) for s, c in zip(a.sample, a.symbol) if c in VENTRICULAR])
        stop = n if onset is None else (onset - SDDB_GAP_S) * FS
        for (s0, r), (e0, _) in zip(notes, notes[1:] + [(n, None)]):
            for s in range(s0, min(e0, stop) - WIN + 1, WIN):
                if len(beats) and ((beats >= s) & (beats < s + WIN)).any():
                    continue
                cat = "NSR" if r == "N" else "other non-shockable"
                yield x[s:s + WIN], dict(db="sddb", record=rec, patient=pid, start_s=s / FS, category=cat,
                                         shock=SHOCK[cat], subtype=r)


def cap_per_person(meta, cap):
    """Keep at most `cap` clips per (person, category), spread evenly over the recording."""
    keep = []
    for _, g in meta.groupby(["patient", "category"]):
        idx = g.sort_values("start_s").index.values
        keep.extend(idx if len(idx) <= cap else idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)])
    return np.sort(np.array(keep))


def build_clips_v2():
    """v1 sources plus CinC 2015 and SDDB, capped per person; writes data/clips.npz (+ data/clips_meta.csv)."""
    X, meta = [], []
    for db in DATABASES:  # v1 loop, unchanged
        for hea in sorted((RAW / db).glob("*.hea")):
            rec = hea.stem
            sig, f = wfdb.rdsamp(str(RAW / db / rec))
            if LEAD[db] not in f["sig_name"]:
                continue
            raw = sig[:, f["sig_name"].index(LEAD[db])]
            track = rhythm_track(db, rec, len(raw))
            x = preprocess(raw, f["fs"])
            track = track[(np.arange(len(x)) * f["fs"] / FS).astype(int)]
            pid = SAME_PATIENT.get(f"{db}/{rec}", f"{db}/{rec}")
            for s in range(0, len(x) - WIN + 1, WIN):
                g = set(track[s:s + WIN])
                if len(g) != 1 or None in g:
                    continue
                cat = aha_category(g.pop(), x[s:s + WIN])
                if cat is None:
                    continue
                X.append(x[s:s + WIN].astype(np.float32))
                meta.append(dict(db=db, record=rec, patient=pid, start_s=s / FS, category=cat, shock=SHOCK[cat],
                                 subtype=""))
        print("read", db, flush=True)
    for gen, name in [(cinc_clips, "cinc2015"), (sddb_clips, "sddb")]:
        for x, m in gen():
            X.append(np.asarray(x, np.float32))
            meta.append(m)
        print("read", name, flush=True)
    meta = pd.DataFrame(meta)
    X = np.stack(X)
    np.savez_compressed(CLIPS.with_name("clips_all.npz"), X=X, **{c: meta[c].values for c in meta})  # uncapped
    before = table(meta)
    keep = cap_per_person(meta, CAP)
    X, meta = X[keep], meta.loc[keep].reset_index(drop=True)
    CLIPS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIPS, X=X, **{c: meta[c].values for c in meta})
    meta.to_csv(CLIPS.with_name("clips_meta.csv"), index=False)
    print("\nBEFORE the per-person cap\n" + before.to_string())
    print(f"\nAFTER capping at {CAP} clips per person per rhythm\n" + table(meta).to_string())
    print(f"\n{len(meta)} clips from {meta.patient.nunique()} people -> {CLIPS}")


def table(meta):
    t = meta.pivot_table(index="category", columns="db", values="start_s", aggfunc="count", fill_value=0)
    t["clips"] = t.sum(axis=1)
    t["people"] = meta.groupby("category")["patient"].nunique()
    t = t.reindex(list(SHOCK)).fillna(0).astype(int)
    tot = meta.groupby(meta.shock.map({1: "SHOCK total", 0: "NO-SHOCK total", -1: "in-between total"}))
    extra = pd.DataFrame({"clips": tot.size(), "people": tot["patient"].nunique()})
    return pd.concat([t, extra]).fillna(0).astype(int)


def load_clips():
    """Clips as a dict of arrays; X is clipped to +-5 mV and given a channel axis, ready for Conv1D."""
    d = dict(np.load(CLIPS, allow_pickle=True))
    d["X"] = np.clip(d["X"], -5, 5)[..., None].astype(np.float32)
    return d


if __name__ == "__main__":
    download()
    build_clips_v2()
