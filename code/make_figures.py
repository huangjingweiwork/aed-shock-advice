"""Step 4: draw the report figures from the results.

    python code/make_figures.py          (after summarise.py)
Writes results/figures/fig*.png.
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from config import CLIPS, AHA, FS, FLASH_BYTES
from compress import l1

RESULTS = config.RESULTS.parent / "results_quick" if "--quick" in sys.argv else config.RESULTS
FIG = RESULTS / "figures"
# Validated categorical slots (blue, orange, aqua, yellow) + ink colours; light theme for print
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 9.5, "axes.titleweight": "bold"})
NICE = {"small": "Small (trained small)", "large": "Large", "large_p75": "Large, pruned to 75%",
        "large_p50": "Large, pruned to 50%", "large_p35": "Large, pruned to 35%",
        "scratch_k50": "50%-width, from scratch", "scratch_k35": "35%-width, from scratch",
        "large_u75": "Large, 75% weights zeroed"}
SHORT = {"small": "Small", "large": "Large", "large_p75": "Pruned 75%", "large_p50": "Pruned 50%",
         "large_p35": "Pruned 35%", "scratch_k50": "Scratch 50%", "scratch_k35": "Scratch 35%",
         "large_u75": "Zeroed 75%"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def fig_examples(z):
    """One 5 s clip of each AHA category, same scale, so amplitude and rate differences are visible."""
    cats = ["coarse VF", "rapid VT", "NSR", "other non-shockable", "asystole"]
    fig, axes = plt.subplots(len(cats), 1, figsize=(6.5, 6), sharex=True, sharey=True)
    t = np.arange(z["X"].shape[1]) / FS
    for ax, c in zip(axes, cats):
        idx = np.flatnonzero(z["category"] == c)
        amp = np.ptp(z["X"][idx], axis=1)
        i = idx[np.argsort(amp)[len(idx) // 2]]  # a typical clip: median peak-to-peak amplitude
        ax.plot(t, z["X"][i], color=C1, lw=1)
        shock = "shock" if c in ("coarse VF", "rapid VT") else "no shock"
        ax.set_title(f"{c}  ({shock})  -  {z['patient'][i]}", loc="left", fontweight="normal", color=INK)
        ax.set_ylim(-3, 3)
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    save(fig, "fig1_example_clips.png")


def fig_quantisation(raw, summary):
    """(a) one layer's weights and the int8 grid; (b) float32 vs int8 output for the same test clips."""
    import keras
    m = keras.models.load_model(RESULTS / "models" / "large_fp32.keras")
    w = m.get_layer("b2a").kernel.numpy()[..., 0].ravel()  # all weights of output filter 0 in layer b2a
    step = np.abs(w).max() / 127  # symmetric per-channel int8: one step = max|w| / 127
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3))
    a.hist(w, bins=60, color=C1, edgecolor="white", linewidth=0.5)
    a.set_title("(a) 288 weights of one filter", loc="left")
    a.set_xlabel("weight value")
    a.set_ylabel("count")
    a.text(0.02, 0.97, f"int8 step = max|w| / 127 = {step:.5f}", transform=a.transAxes, va="top", color=INK,
           fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    a.set_ylim(0, a.get_ylim()[1] * 1.2)
    te = np.concatenate([r["test_idx"] for r in raw])
    p32 = np.concatenate([r["large_fp32__test"] for r in raw])
    p8 = np.concatenate([r["large_int8__test"] for r in raw])
    b.scatter(p32, p8, s=8, color=C1, alpha=0.35, edgecolors="none")
    b.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    flips = int(((p32 >= 0.5) != (p8 >= 0.5)).sum())
    b.set_title("(b) Large model: float32 vs int8 output", loc="left")
    b.set_xlabel("P(shock), float32")
    b.set_ylabel("P(shock), int8")
    b.text(0.03, 0.95, f"{len(te):,} test clips\n{flips} decisions flip at 0.5", transform=b.transAxes, va="top",
           color=INK2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_quantisation.png")


def fig_pruning_l1():
    """Sorted filter L1 norms in one layer, with the filters a 50% prune keeps."""
    import keras
    m = keras.models.load_model(RESULTS / "models" / "large_fp32.keras")
    s = np.sort(l1(m.get_layer("b4a").kernel.numpy()))[::-1]
    keep = len(s) // 2
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(np.arange(keep), s[:keep], color=C1, width=0.85, label=f"kept ({keep})")
    ax.bar(np.arange(keep, len(s)), s[keep:], color="#c3c2b7", width=0.85, label=f"removed ({len(s) - keep})")
    ax.set_xlabel("filter, sorted by L1 norm")
    ax.set_ylabel("L1 norm / layer mean")
    ax.set_title("Structured pruning at 50%: layer b4a keeps its 64 strongest of 128 filters", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig4_pruning_l1.png")


def fig_pruning_recovery(summary):
    """AUROC straight after cutting vs after fine-tuning, per pruning level."""
    s = summary.set_index("model")["auroc"]
    keeps = [100, 75, 50, 35]
    noft = [s["large_fp32"]] + [s[f"large_p{k}_noft_fp32"] for k in keeps[1:]]
    ft = [s["large_fp32"]] + [s[f"large_p{k}_fp32"] for k in keeps[1:]]
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.plot(keeps, noft, color=C2, lw=2, marker="o", ms=6, label="cut, no retraining")
    ax.plot(keeps, ft, color=C1, lw=2, marker="o", ms=6, label="cut + 20 epochs fine-tuning")
    for x, y in zip(keeps[1:], noft[1:]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -10), color=INK2, fontsize=8)
    ax.invert_xaxis()
    ax.set_xticks(keeps)
    ax.set_xlabel("% of filters kept in every layer")
    ax.set_ylabel("AUROC (shock vs no shock)")
    ax.set_ylim(min(0.4, min(noft) - 0.05), 1.02)
    ax.set_title("Pruning damages the model; fine-tuning repairs it", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_pruning_recovery.png")


def fig_headline(summary):
    """AHA rows passed vs int8 file size: compressed-large models vs models trained small."""
    s = summary[summary.stage == "int8"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    groups = [("compressed from the large model", ["large", "large_p75", "large_p50", "large_p35", "large_u75"], C1, "o"),
              ("trained small from the start", ["scratch_k50", "scratch_k35", "small"], C2, "s")]
    for label, fams, col, mk in groups:
        g = s[s.family.isin(fams)].copy()
        # pruned and scratch twins (and large / zeroed) share a file size: nudge them apart so both show
        g["file_kib"] *= g.family.map({"large_p50": 0.94, "large_p35": 0.94, "scratch_k50": 1.06, "scratch_k35": 1.06,
                                       "large": 0.94, "large_u75": 1.06}).fillna(1.0)
        ax.scatter(g.file_kib, g["passed_0.5"], s=60, color=col, marker=mk, label=label, zorder=3,
                   edgecolors="white", linewidths=1.5)
        for _, r in g.iterrows():  # short labels; points sharing a size are split above / below
            dy = {"large_u75": -13, "scratch_k35": -13, "scratch_k50": -13, "large_p75": -13}.get(r.family, 9)
            ax.annotate(SHORT[r.family], (r.file_kib, r["passed_0.5"]), textcoords="offset points",
                        xytext=(-4 if dy > 0 else 4, dy), fontsize=7.5, color=INK2, va="center",
                        ha="right" if dy > 0 else "left")
    ax.axvline(FLASH_BYTES / 1024, color=INK, lw=1, ls="--")
    ax.text(FLASH_BYTES / 1024 * 0.93, 5.25, "chip flash: 256 KiB", ha="right", color=INK, fontsize=8)
    ax.set_xscale("log")
    ticks = [10, 25, 50, 100, 250, 500, 1000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(8, 1500)
    ax.set_xlabel("int8 model file (KiB, log scale)")
    ax.set_ylabel("AHA targets passed (of 5)")
    ax.set_ylim(-0.3, 5.5)
    ax.set_yticks(range(6))
    ax.set_title("Which int8 models fit the chip, and how many AHA targets they meet", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig7_headline.png")


def fig_per_category(aha):
    """Score per AHA category for the main contenders, against each goal."""
    models = [("large_fp32", "Large, float32 (reference, too big)", C1), ("large_p35_int8", "Large pruned to 35% + int8", C3),
              ("scratch_k35_int8", "35%-width from scratch + int8", C2), ("small_int8", "Small + int8", C4)]
    cats = [c for c, *_ in AHA]
    t = aha[aha.threshold.astype(str) == "0.5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.19
    for j, (m, label, col) in enumerate(models):
        r = t[t.model == m].set_index("category").reindex(cats)
        ax.bar(np.arange(len(cats)) + (j - 1.5) * w, r["score"], width=w - 0.02, color=col, label=label)
    for i, (c, metric, goal, _) in enumerate(AHA):
        ax.plot([i - 2.1 * w, i + 2.1 * w], [goal, goal], color=INK, lw=1.5)
    ax.plot([], [], color=INK, lw=1.5, label="AHA goal")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n({'Se' if m == 'sensitivity' else 'Sp'})" for c, m, *_ in AHA], fontsize=8)
    low = t[t.model.isin([m for m, *_ in models]) & t.category.isin(cats)]["score"].min()
    ax.set_ylim(max(0.0, min(0.5, np.floor((low - 0.05) * 10) / 10)), 1.02)
    ax.set_ylabel("score (threshold 0.5)")
    ax.set_title("Score per AHA rhythm category", loc="left", pad=30)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    save(fig, "fig8_per_category.png")


def fig_structured_vs_unstructured(summary):
    """Same 75% of weights removed: zeros (unstructured) vs a smaller network (structured)."""
    s = summary.set_index("model")
    rows = [("large_int8", "Large\n(no pruning)"), ("large_u75_int8", "75% of weights\nzeroed (unstructured)"),
            ("large_p50_int8", "50% of filters removed\n(structured, ~75% fewer weights)")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.2, 3))
    ax.bar(x - 0.2, [s.loc[m, "file_kib"] for m, _ in rows], 0.38, color=C1, label="int8 file (what the chip stores)")
    ax.bar(x + 0.2, [s.loc[m, "gzip_kib"] for m, _ in rows], 0.38, color=C3, label="same file gzipped (disk only)")
    for i, (m, _) in enumerate(rows):
        ax.text(i - 0.2, s.loc[m, "file_kib"] + 8, f"{s.loc[m, 'file_kib']:.0f}", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, s.loc[m, "gzip_kib"] + 8, f"{s.loc[m, 'gzip_kib']:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(FLASH_BYTES / 1024, color=INK, lw=1, ls="--")
    ax.text(2.45, FLASH_BYTES / 1024 + 8, "256 KiB flash", ha="right", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in rows], fontsize=8)
    ax.set_ylabel("KiB")
    ax.set_title("Zeros only shrink a compressed file; removing filters shrinks the model", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_structured_vs_unstructured.png")


def fig_pipeline():
    """Boxes-and-arrows overview of the pipeline, with the tool used at each step."""
    from matplotlib.patches import FancyBboxPatch
    steps = [("1. Data", "PhysioNet CUDB, VFDB,\nMIT-BIH → 5 s clips,\n125 Hz, AHA labels", "wfdb, SciPy"),
             ("2. Train", "large (501k weights)\nsmall (4.6k weights)\npatient-wise 5 folds", "TensorFlow / Keras"),
             ("3. Prune", "remove lowest-L1 filters\n(keep 75/50/35%)\nthen fine-tune", "NumPy + Keras"),
             ("4. Quantise", "float32 → int8\ncalibrate on 300\ntraining clips", "TFLite converter"),
             ("5. Score", "run the .tflite file\nper AHA category,\nsize, RAM, speed", "TFLite interpreter")]
    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, body, tool) in enumerate(steps):
        x = i + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.12), 0.8, 0.8, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor="#eef4fc", edgecolor=C1, linewidth=1.2))
        ax.text(x + 0.4, 0.83, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + 0.4, 0.66, body, ha="center", va="top", fontsize=7, color=INK, linespacing=1.3)
        ax.text(x + 0.4, 0.2, tool, ha="center", va="center", fontsize=6.5, color=C1, style="italic")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.05, 0.52), xytext=(x + 0.81, 0.52),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    save(fig, "fig2_pipeline.png")


def fig_asystole_input(z):
    """An asystole clip before and after the int8 input rounding (step = input scale of the .tflite file)."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(RESULTS / "models" / "large_p35_int8.tflite"))
    sc, zp = it.get_input_details()[0]["quantization"]
    idx = np.flatnonzero(z["category"] == "asystole")
    x = z["X"][idx[np.argsort(np.ptp(z["X"][idx], axis=1))[len(idx) // 2]]]
    q = (np.clip(np.round(x / sc + zp), -128, 127) - zp) * sc
    t = np.arange(len(x)) / FS
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.plot(t, 1000 * x, color=MUTED, lw=1, label="float32 input")
    ax.step(t, 1000 * q, color=C2, lw=1.4, where="mid", label=f"int8 input (step = {1000 * sc:.0f} µV)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("µV")
    ax.set_title("Asystole after int8 rounding: a flat line becomes a few integer levels", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig9_asystole_input.png")


def main():
    fig_pipeline()
    z = np.load(CLIPS, allow_pickle=True)
    summary = pd.read_csv(RESULTS / "tables" / "summary.csv")
    aha = pd.read_csv(RESULTS / "tables" / "aha_by_model.csv")
    raw = [np.load(f) for f in sorted((RESULTS / "raw").glob("fold*.npz"))]
    fig_examples(z)
    fig_quantisation(raw, summary)
    fig_pruning_l1()
    fig_pruning_recovery(summary)
    fig_headline(summary)
    fig_per_category(aha)
    fig_structured_vs_unstructured(summary)
    fig_asystole_input(z)


if __name__ == "__main__":
    main()
