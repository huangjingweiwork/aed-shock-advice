"""Step 3: turn the per-fold predictions into result tables.

    python code/summarise.py

Every test clip was predicted by a model that never saw that patient, so all 5 folds are pooled and scored
together. Each model is scored twice: with the decision threshold fixed at 0.5, and with a threshold chosen
per fold on that fold's validation patients (never on the test patients).
Writes results/tables/*.csv, results/thresholds.json and results/RESULTS.md.
"""
import json, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import config
from config import CLIPS, AHA, FLASH_BYTES, RAM_BYTES, MCU_HZ, MACS_PER_CYCLE
from compress import latency_ms
from evaluate import aha_table, choose_threshold

STAGES = ["fp32", "dyn", "noft_fp32", "int8"]
FAMILIES = ["small", "large", "large_p75", "large_p50", "large_p35", "scratch_k50", "scratch_k35", "large_u75"]
SHOWN = ["coarse VF", "rapid VT", "NSR", "other non-shockable", "asystole", "fine VF", "other VT"]


def split_name(name):
    fam = re.match(r"(small|large_p\d+|large_u\d+|large|scratch_k\d+)_(.*)", name)
    return fam.group(1), fam.group(2)


def md_table(df):
    """DataFrame -> Markdown table (no extra packages)."""
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


RESULTS = config.RESULTS.parent / "results_quick" if "--quick" in sys.argv else config.RESULTS


def by_source(z, prob, out, models=("large_fp32", "scratch_k50_fp32", "scratch_k50_int8", "small_int8")):
    """Share of clips decided correctly at threshold 0.5, by source database and shock / no-shock (the by-source table in the report).
    CinC 2015 alarms are split into true VF, other true alarms and false alarms; in-between rhythms are left out."""
    db, shock = z["db"], z["shock"]
    sub = z["subtype"] if "subtype" in z.files else np.full(len(db), "", object)
    name = {"cudb": "CUDB", "vfdb": "VFDB", "mitdb": "MIT-BIH", "sddb": "SDDB", "cinc2015": "CinC 2015"}
    group = np.full(len(db), "", object)
    for i in np.flatnonzero(shock >= 0):
        if db[i] == "cinc2015":
            group[i] = ("CinC 2015: false alarms (no shock)" if str(sub[i]).startswith("false") else
                        "CinC 2015: true VF (shock)" if shock[i] == 1 else
                        "CinC 2015: true asystole / slow / fast (no shock)")
        else:
            group[i] = f"{name[db[i]]}: {'shockable' if shock[i] == 1 else 'non-shockable'}"
    rows = []
    for g in sorted(set(group) - {""}):
        m = (group == g) & ~np.isnan(prob[models[0]])
        rows.append(dict(group=g, clips=int(m.sum()),
                         **{n: float(((prob[n][m] >= 0.5) == (shock[m] == 1)).mean()) for n in models if n in prob}))
    pd.DataFrame(rows).to_csv(out / "by_source.csv", index=False)


def main():
    z = np.load(CLIPS, allow_pickle=True)
    cat, pat, shock = z["category"], z["patient"], z["shock"]
    folds = sorted((RESULTS / "raw").glob("fold*.npz"))
    sizes = json.load(open(RESULTS / "sizes.json"))
    raw = [np.load(f) for f in folds]
    names = sorted({k.split("__")[0] for k in raw[0].files if "__" in k},
                   key=lambda n: (FAMILIES.index(split_name(n)[0]), STAGES.index(split_name(n)[1])))
    print(f"{len(folds)} folds, {len(names)} model variants")

    prob, thr_at, thresholds = {}, {}, {}
    for name in names:
        p, t = np.full(len(cat), np.nan, np.float32), np.full(len(cat), np.nan, np.float32)
        thresholds[name] = []
        for r in raw:
            va, te = r["val_idx"], r["test_idx"]
            th = choose_threshold(cat[va], r[f"{name}__val"])
            thresholds[name].append(th)
            p[te], t[te] = r[f"{name}__test"], th
        prob[name], thr_at[name] = p, t
    done = ~np.isnan(prob[names[0]])

    tables, rows = [], []
    for name in names:
        fam, stage = split_name(name)
        row = dict(model=name, family=fam, stage=stage)
        s = sizes.get(name, {})
        row.update(params=s.get("params"), macs=s.get("macs"), sparsity=s.get("sparsity"), file_kib=s.get("file_kib"),
                   gzip_kib=s.get("gzip_kib"), peak_ram_kib=s.get("peak_ram_kib"))
        f = RESULTS / "models" / f"{name}.tflite"
        row["pc_ms"] = latency_ms(f.read_bytes(), np.zeros((625, 1), np.float32)) if f.exists() else None
        if stage == "int8":
            row["mcu_ms_low"] = 1000 * s["macs"] / (MACS_PER_CYCLE[1] * MCU_HZ)
            row["mcu_ms_high"] = 1000 * s["macs"] / (MACS_PER_CYCLE[0] * MCU_HZ)
            row["fits_chip"] = bool(s["file_kib"] * 1024 <= FLASH_BYTES and s["peak_ram_kib"] * 1024 <= RAM_BYTES)
        both = done & (shock >= 0)
        row["auroc"] = roc_auc_score(shock[both], prob[name][both])
        for mode, shocked in (("0.5", prob[name] >= 0.5), ("tuned", prob[name] >= thr_at[name])):
            tab = aha_table(cat[done], pat[done], shocked[done])
            tab.insert(0, "threshold", mode)
            tab.insert(0, "model", name)
            tables.append(tab)
            graded = tab[tab["passes"].notna()]
            row[f"sens_all_{mode}"] = float(shocked[done & (shock == 1)].mean())   # all shockable clips
            row[f"spec_all_{mode}"] = float((~shocked[done & (shock == 0)]).mean())  # all non-shockable clips
            row[f"passed_{mode}"] = int(graded["passes"].sum())
            row[f"worst_margin_{mode}"] = float((graded["score"] - graded["goal"]).min())
            if mode == "tuned":
                row.update({c: v for c, v in zip(tab["category"], tab["score"])})
        row["threshold_mean"] = float(np.mean(thresholds[name]))
        own = prob.get(f"{fam}_fp32")
        row["flips_vs_own_fp32"] = int(((prob[name] >= 0.5) != (own >= 0.5))[done].sum()) if own is not None else None
        row["flips_vs_large_fp32"] = int(((prob[name] >= 0.5) != (prob["large_fp32"] >= 0.5))[done].sum())
        rows.append(row)

    out = RESULTS / "tables"
    out.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(tables).to_csv(out / "aha_by_model.csv", index=False)
    json.dump(thresholds, open(RESULTS / "thresholds.json", "w"), indent=1)
    meta = pd.DataFrame(dict(category=cat, patient=pat, db=z["db"]))
    data = meta.pivot_table(index="category", columns="db", values="patient", aggfunc="count", fill_value=0)
    data["clips"] = data.sum(axis=1)
    data["patients"] = meta.groupby("category")["patient"].nunique()
    data = data.reindex(SHOWN).reset_index()
    data.to_csv(out / "dataset.csv", index=False)
    by_source(z, prob, out)

    # ---- RESULTS.md: the human-readable version
    goals = {c: f"{'Se' if m == 'sensitivity' else 'Sp'} > {g:.0%}" for c, m, g, _ in AHA}
    main_cols = ["model", "params", "file_kib", "peak_ram_kib", "fits_chip", "auroc", "sens_all_0.5", "spec_all_0.5",
                 "sens_all_tuned", "spec_all_tuned", "passed_0.5", "passed_tuned",
                 "worst_margin_tuned"] + [c for c in SHOWN if c in summary]
    md = ["# Results", "", "Generated by `code/summarise.py` from `results/raw/`. Pooled out-of-fold predictions from "
          f"patient-wise {len(folds)}-fold cross-validation.", "",
          "## Dataset", "", md_table(data), "",
          "## All models", "",
          "`passed_*` = AHA rows passed out of 5 (score beats the goal and the lower 90% bound reaches the AHA minimum). "
          "Per-category columns are scores at the tuned threshold. AHA goals: "
          + "; ".join(f"{c} {g}" for c, g in goals.items()) + ".", "",
          md_table(summary[main_cols]), "",
          "## Size, memory and speed (int8 files)", "",
          f"Chip budget: {FLASH_BYTES // 1024} KiB flash, {RAM_BYTES // 1024} KiB RAM, {MCU_HZ / 1e6:.0f} MHz. "
          "`mcu_ms` is an estimate from the MAC count at 0.5-1 MAC per cycle; `pc_ms` is measured on a laptop CPU.", "",
          md_table(summary[summary.stage == "int8"][["model", "params", "macs", "file_kib", "gzip_kib", "peak_ram_kib",
                                                    "fits_chip", "mcu_ms_low", "mcu_ms_high", "pc_ms"]]), "",
          "## Decision thresholds chosen on validation patients (per fold)", "",
          md_table(pd.DataFrame([dict(model=n, **{f"fold{i}": t for i, t in enumerate(v)}) for n, v in thresholds.items()]))]
    (RESULTS / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary[main_cols].to_string(index=False))


if __name__ == "__main__":
    main()
