"""
Describes the data the full run trains on (after rrg.text.clean_target and
--mimic_repeat), next to the test sets, so the run's choices can be checked
before spending days of GPU time. CPU only. --model picks the tokenizer for the
token-length plot (rrg/models.py); everything else is model-independent.

    python tools/analyze_data.py                    # -> outputs/data_analysis/*.png + ringkasan.json
    python tools/analyze_data.py --model chexone --sec_per_step 12
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rrg_indo/, for the rrg package
from rrg.config import MANIFEST_PATH, OUTPUT_DIR, TRAIN_SPLITS, TASKS  # noqa: E402
from rrg.models import DEFAULT_MODEL, MODELS  # noqa: E402
from rrg.prompts import build_prompt  # noqa: E402
from rrg.text import ANON_BLANK, CONSULT_PHONE, DEGENERATE, clean_target  # noqa: E402

OUT_DIR = OUTPUT_DIR / "data_analysis"
TASK_ID = {"findings": "TEMUAN", "impression": "KESAN"}

# reference palette (dataviz skill, light mode): categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# the four text populations compared throughout; test references are as in the manifest
GROUPS = [("CheXpert (train)", "train_internal"), ("MIMIC (train)", "train_mimic"),
          ("MIMIC (test)", "test_mimic"), ("OpenI (test)", "test_openi")]

PRIOR_WORDS = re.compile(r"\b(stabil|persisten|tidak berubah|dibandingkan|sebelumnya|interval)\b")
NUMBERED = re.compile(r"^\s*1\.\s")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "legend.frameon": False, "font.size": 10, "figure.dpi": 130,
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] {name}")


def texts(df, task, clean):
    x = df[task].dropna()
    x = x[x.str.len() > 0]
    return x.map(clean_target).dropna() if clean else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS), help="tokenizer for the token-length plot")
    p.add_argument("--mimic_repeat", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--sec_per_step", type=float, default=19.0, help="measured on the MedGemma 40k run")
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()
    spec = MODELS[args.model]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    m = pd.read_parquet(MANIFEST_PATH)
    train = m[m["split"].isin(TRAIN_SPLITS)]
    summary = {}

    # ---- 1. sample composition before / after MIMIC oversampling ----------------------------
    counts = {(src, t): len(texts(train[train["source"].eq(src)], t, clean=True))
              for src in ["chexpert_plus", "mimic_cxr"] for t in TASKS}
    raw_total = sum(len(texts(train, t, clean=False)) for t in TASKS)
    comp = pd.DataFrame({
        "Sebelum (MIMIC ×1)": [counts[("chexpert_plus", t)] for t in TASKS] + [counts[("mimic_cxr", t)] for t in TASKS],
        f"Sesudah (MIMIC ×{args.mimic_repeat})": [counts[("chexpert_plus", t)] for t in TASKS]
                                               + [counts[("mimic_cxr", t)] * args.mimic_repeat for t in TASKS],
    }, index=[f"CheXpert {TASK_ID[t]}" for t in TASKS] + [f"MIMIC {TASK_ID[t]}" for t in TASKS]).T
    total = int(comp.iloc[1].sum())
    steps = int(np.ceil(total / (args.batch_size * args.grad_accum)))
    summary["komposisi"] = {"mentah_sebelum_clean": raw_total, **{k: {c: int(v) for c, v in r.items()}
                                                                  for k, r in comp.iterrows()}}
    summary["estimasi_run"] = {"sampel": total, "step": steps, "jam": round(steps * args.sec_per_step / 3600, 1)}

    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = np.zeros(len(comp))
    for i, col in enumerate(comp.columns):
        vals = comp[col].to_numpy()
        ax.barh(comp.index, vals, left=left, color=SERIES[i], height=0.5, edgecolor=SURFACE, linewidth=2, label=col)
        for y, (l, v) in enumerate(zip(left, vals)):
            share = v / comp.iloc[y].sum()
            if share > 0.04:
                ax.text(l + v / 2, y, f"{share:.0%}", ha="center", va="center", color="white", fontsize=9,
                        fontweight="bold")
        left += vals
    for y, s in enumerate(left):
        ax.text(s * 1.01, y, f"{int(s):,}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, left.max() * 1.15)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_title("Komposisi sampel training (setelah pembersihan)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.15), fontsize=9)
    save(fig, "01_komposisi_sampel.png")

    # ---- 2. images per split ------------------------------------------------------------------
    split_order = ["train_internal", "dev_internal", "train_mimic", "test_openi", "test_mimic", "test_chexpert"]
    role = {"train_internal": "train", "train_mimic": "train", "dev_internal": "eval_loss",
            "test_chexpert": "test", "test_mimic": "test", "test_openi": "test eksternal"}
    n_split = m["split"].value_counts().reindex(split_order)
    summary["gambar_per_split"] = {k: int(v) for k, v in n_split.items()}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh([f"{s}  ({role[s]})" for s in split_order], n_split.values, color=SERIES[0], height=0.55)
    for y, v in enumerate(n_split.values):
        ax.text(v + n_split.max() * 0.01, y, f"{v:,}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, n_split.max() * 1.12)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_title("Jumlah gambar per split")
    save(fig, "02_gambar_per_split.png")

    # ---- 3. target length (words) per population ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    summary["panjang_kata_median"] = {}
    bins = np.arange(0, 162, 4)
    for ax, t in zip(axes, TASKS):
        for i, (name, split) in enumerate(GROUPS):
            x = texts(m[m["split"].eq(split)], t, clean=split in TRAIN_SPLITS).str.split().str.len()
            med = int(x.median())
            summary["panjang_kata_median"][f"{TASK_ID[t]} {name}"] = med
            h, e = np.histogram(x.clip(upper=bins[-1] - 1), bins=bins, density=True)
            ax.plot(e[:-1] + 2, h * 100, color=SERIES[i], linewidth=2, label=f"{name}, median {med}")
        ax.set_title(f"Panjang {TASK_ID[t]} (kata)")
        ax.set_xlabel("jumlah kata (≥160 digabung di ujung)")
        ax.set_ylabel("% sampel per 4 kata")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5)
    save(fig, "03_panjang_target_kata.png")

    # ---- 4. full sequence length in tokens vs max_length ---------------------------------------
    # approximate (prompt + ~12 chat-template tokens); tools/check_model.py measures it exactly
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    prompt_len = {t: len(tok(build_prompt(t, "AP")).input_ids) + 12 for t in TASKS}
    seq = []
    for t in TASKS:
        x = texts(train, t, clean=True).tolist()
        ids = tok(x, add_special_tokens=False).input_ids
        seq.append(np.array([len(i) for i in ids]) + spec.image_tokens + prompt_len[t])
    seq = np.concatenate(seq)
    n_over = int((seq > args.max_length).sum())
    over = n_over / len(seq)
    summary["panjang_token"] = {"model": spec.key, "median": int(np.median(seq)), "p95": int(np.percentile(seq, 95)),
                                "p99": int(np.percentile(seq, 99)), "max": int(seq.max()),
                                "jumlah_terpotong": n_over, "persen_terpotong": round(over * 100, 3)}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(np.clip(seq, 0, 1200), bins=np.arange(250, 1210, 10), color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(args.max_length, color=INK, linewidth=1.2, linestyle="--")
    ax.text(args.max_length - 8, ax.get_ylim()[1] * 0.9, f"max_length {args.max_length}\n{n_over} sampel terpotong ({over:.3%})",
            ha="right", va="top", fontsize=9, color=INK)
    ax.axvline(np.median(seq), color=INK_2, linewidth=1, linestyle=":")
    ax.text(np.median(seq) + 8, ax.get_ylim()[1] * 0.9, f"median {int(np.median(seq))}", fontsize=9, color=INK_2,
            va="top")
    ax.set_title(f"Panjang sekuens training, {spec.key} ({spec.image_tokens} token gambar + prompt + target)")
    ax.set_xlabel("token (≥1200 digabung di ujung)")
    ax.set_ylabel("jumlah sampel")
    ax.grid(axis="x", visible=False)
    save(fig, "04_panjang_token.png" if spec.key == DEFAULT_MODEL else f"04_panjang_token_{spec.key}.png")

    # ---- 5. impression style per population ----------------------------------------------------
    metrics = {"bernomor (\"1. …\")": lambda s: NUMBERED.search(s) is not None,
               "menyebut pembanding\n(stabil/persisten/sebelumnya…)": lambda s: PRIOR_WORDS.search(s) is not None,
               "≤ 5 kata": lambda s: len(s.split()) <= 5}
    style_tab = pd.DataFrame({name: {k: texts(m[m["split"].eq(split)], "impression",
                                              clean=split in TRAIN_SPLITS).map(f).mean() * 100
                                     for k, f in metrics.items()}
                              for name, split in GROUPS})
    summary["gaya_kesan_persen"] = {c.replace("\n", " "): {k.replace("\n", " "): round(v, 1)
                                                          for k, v in style_tab[c].items()}
                                    for c in style_tab.columns}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.19
    x0 = np.arange(len(style_tab))
    for i, col in enumerate(style_tab.columns):
        bars = ax.bar(x0 + (i - 1.5) * (w + 0.01), style_tab[col], width=w, color=SERIES[i], label=col)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x0, style_tab.index)
    ax.set_ylabel("% laporan KESAN")
    ax.set_ylim(0, style_tab.values.max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Gaya KESAN: CheXpert vs MIMIC vs OpenI")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
    save(fig, "05_gaya_kesan.png")

    # ---- 6. what clean_target removes ----------------------------------------------------------
    raw = pd.concat([texts(train, t, clean=False) for t in TASKS])
    cleaned = raw.map(clean_target)
    rules = {
        "nomor telepon konsultasi": raw.str.contains(CONSULT_PHONE),
        "kalimat attestation\n(\"saya telah meninjau…\")": raw.str.contains(r"saya (?:telah|sudah) meninjau"),
        "kalimat komunikasi\n(\"dikomunikasikan kepada…\")": raw.str.contains(
            r"dikomunikasikan|didiskusikan dengan|diberitahukan|melalui telepon|via telepon"),
        "blank anonimisasi \"___\"": raw.str.contains(ANON_BLANK),
        "terjemahan berulang (dibuang)": raw.map(lambda s: DEGENERATE.search(s + " ") is not None),
    }
    rule_n = pd.Series({k: int(v.sum()) for k, v in rules.items()})
    summary["pembersihan"] = {**{k.replace("\n", " "): v for k, v in rule_n.items()},
                              "total_target_diubah": int((cleaned.notna() & (cleaned != raw)).sum()),
                              "total_target_dibuang": int(cleaned.isna().sum()), "total_target": len(raw)}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(rule_n.index, rule_n.values, color=SERIES[0], height=0.55)
    for y, v in enumerate(rule_n.values):
        ax.text(v + rule_n.max() * 0.01, y, f"{v:,}  ({v / len(raw):.1%})", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, rule_n.max() * 1.3)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Yang dibersihkan clean_target (dari {len(raw):,} target training)")
    ax.set_xlabel("jumlah target yang mengandung")
    save(fig, "06_pembersihan.png")

    # ---- 7. view (projection) mix per split ----------------------------------------------------
    view_map = {"AP": "AP", "PA": "PA", "LATERAL": "Lateral", "LL": "Lateral", "RL": "Lateral"}
    vsplits = ["train_internal", "train_mimic", "test_chexpert", "test_mimic"]
    vt = (m[m["split"].isin(vsplits)].assign(v=lambda d: d["view"].map(view_map).fillna("tidak diketahui"))
          .groupby("split")["v"].value_counts(normalize=True).unstack(fill_value=0).reindex(vsplits) * 100)
    vt = vt[[c for c in ["AP", "PA", "Lateral", "tidak diketahui"] if c in vt.columns]]
    summary["proyeksi_persen"] = {s: {k: round(v, 1) for k, v in r.items()} for s, r in vt.iterrows()}
    fig, ax = plt.subplots(figsize=(9, 3.0))
    left = np.zeros(len(vt))
    for i, col in enumerate(vt.columns):
        ax.barh(vt.index, vt[col], left=left, color=SERIES[i], height=0.55, edgecolor=SURFACE, linewidth=2, label=col)
        for y, (l, v) in enumerate(zip(left, vt[col])):
            if v > 6:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9,
                        fontweight="bold")
        left += vt[col].to_numpy()
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Proyeksi foto per split (OpenI tidak punya info proyeksi)")
    ax.set_xlabel("% gambar")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=9)
    save(fig, "07_proyeksi.png")

    name = "ringkasan.json" if spec.key == DEFAULT_MODEL else f"ringkasan_{spec.key}.json"
    with open(OUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[analyze] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
