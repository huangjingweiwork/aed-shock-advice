"""Plot -20 s .. +240 s around each SDDB VF onset, to check by eye how long the VF lasts (sets SDDB_VF_S)."""
import sys
sys.path.insert(0, "code")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, wfdb
from config import RAW, FS
from data import SDDB_VF_ONSET, _seconds, preprocess, amplitude_uv, rate_bpm

recs = sorted(SDDB_VF_ONSET)
fig, axes = plt.subplots(len(recs), 1, figsize=(16, 1.25 * len(recs)), sharex=True)
for ax, rec in zip(axes, recs):
    t0 = _seconds(SDDB_VF_ONSET[rec])
    hdr = wfdb.rdheader(str(RAW / "sddb" / rec))
    a, b = max(0, (t0 - 20) * hdr.fs), min(hdr.sig_len, (t0 + 240) * hdr.fs)
    sig, f = wfdb.rdsamp(str(RAW / "sddb" / rec), sampfrom=a, sampto=b, channels=[0])
    x = preprocess(sig[:, 0], f["fs"])
    t = a / hdr.fs - t0 + np.arange(len(x)) / FS
    ax.plot(t, np.clip(x, -3, 3), lw=0.4, color="k")
    ax.axvline(0, color="r"), ax.axvline(120, color="b", ls="--")
    amps = [amplitude_uv(x[i:i + 5 * FS]) for i in range(0, len(x) - 5 * FS, 5 * FS)]
    ax.set_ylabel(rec, rotation=0, labelpad=12)
    ax.set_ylim(-3, 3)
    ax.text(1.002, 0.5, f"{hdr.comments[:1]}", transform=ax.transAxes, fontsize=7, va="center")
axes[-1].set_xlabel("seconds from published VF onset (red); blue = end of the window used")
plt.tight_layout()
plt.savefig("data/sddb_vf_check.png", dpi=110)
print("saved data/sddb_vf_check.png")
for rec in recs:
    print(rec, wfdb.rdheader(str(RAW / "sddb" / rec)).comments)
