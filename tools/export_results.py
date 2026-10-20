"""
Exports the public, text-free results to results/ and draws the figures for the README.

outputs/ stays private: its prediction CSVs, translation caches and per-sample files contain
dataset report text (MIMIC-CXR and CheXpert Plus may not be redistributed). This script copies
only aggregate numbers:

  results/indonesian_metrics.json   Indonesian metrics per run / test set / track (0-1 scale)
  results/rexrank_metrics.json      ReXrank metrics on ID->EN translations + translation ceiling
  results/significance.json         paired-bootstrap differences and 95% CIs (outputs/comparisons/)
  results/leaderboard.json          rexrank.ai snapshot (findings, study level) + our ranks
  results/training_curves.json      eval loss per checkpoint
  results/RESULTS.md                all of the above as tables (the table view of the figures)
  results/figures/*_{light,dark}.png

    uv run python tools/export_results.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = ROOT / "results"
FIG = RES / "figures"

RUNS = {"medgemma_full": "MedGemma-FT", "chexone_full": "CheXOne-FT"}
ZERO_SHOT = "medgemma-1.5-4b-it"  # zero-shot MedGemma, CheXpert only, old metrics file (BLEU on 0-100)
SPLITS = {"test_chexpert": "CheXpert Plus", "test_mimic": "MIMIC-CXR", "test_openi": "IU-Xray"}
TRACKS = {"findings": "Findings", "impression": "Impression", "findings_impression": "F+I"}
LEADERBOARD = OUT / "rexrank_leaderboard" / "rexrank_tables_2026-10-16.json"
LB_COLS = ["CRIMSON", "1/RadCliQ-v1", "BLEU", "BertScore", "SembScore", "RadGraph", "RaTEScore", "GREEN"]
LB_KEY = {"1/RadCliQ-v1": "1/radcliq_v1", "RaTEScore": "ratescore", "GREEN": "green", "CRIMSON": "crimson",
          "RadGraph": "radgraph", "SembScore": "semb_score", "BLEU": "bleu2", "BertScore": "bertscore"}
LB_DS = {"test_chexpert": "CheXpert", "test_mimic": "MIMIC", "test_openi": "IU"}

# reference palette of the dataviz method, validated for these two series in both modes
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", medgemma_full="#2a78d6", chexone_full="#eb6834", baseline="#b4b2aa"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", medgemma_full="#3987e5", chexone_full="#d95926", baseline="#6b6a66"),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ data
def indonesian_metrics() -> dict:
    out = {}
    for run in RUNS:
        out[run] = {s: load(OUT / run / "metrics" / f"{run}__{s}_metrics.json")[s] for s in SPLITS}
    zs = load(OUT / "metrics" / f"{ZERO_SHOT}__test_chexpert_metrics.json")["test_chexpert"]
    out["medgemma_zeroshot"] = {"test_chexpert": {
        t: {k: (v / 100 if k.startswith("bleu_") else v) for k, v in m.items()} for t, m in zs.items()}}
    return out


def rexrank_metrics() -> dict:
    out = {"translator": None, "llm_n": None, "runs": {}}
    for run in RUNS:
        d = load(OUT / run / "rexrank" / f"{run}__rexrank.json")
        out.update(translator=d["translator"], llm_n=d["llm_n"])
        out["runs"][run] = d["results"]
    out["runs"]["reference_backtranslation"] = load(
        OUT / "rexrank_calibration" / "reference_backtranslation__rexrank.json")["results"]
    return out


def significance() -> dict:
    """Paired-bootstrap B-A (A = medgemma_full, B = chexone_full) with 95% CI, from outputs/comparisons/."""
    out = {"A": "medgemma_full", "B": "chexone_full", "indonesian": {}, "rexrank": {}}
    files = [("rexrank", OUT / "comparisons" / "rexrank_medgemma_full_vs_chexone_full.txt")] + [
        ("indonesian", OUT / "comparisons" / f"indonesian_{s}_medgemma_full_vs_chexone_full.txt") for s in SPLITS]
    for family, path in files:
        key = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"=== (\S+) (\S+)", line)
            if m:
                key = f"{m.group(1)}/{m.group(2)}"
                continue
            m = re.match(r"(\S+(?: corpus)?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]", line)
            if m and key:
                lo, hi = float(m.group(5)), float(m.group(6))
                out[family].setdefault(key, {})[m.group(1).replace(" corpus", "")] = {
                    "A": float(m.group(2)), "B": float(m.group(3)), "B_minus_A": float(m.group(4)),
                    "ci95": [lo, hi], "significant": lo > 0 or hi < 0}
    return out


def leaderboard(rex: dict) -> dict:
    lb = load(LEADERBOARD)
    out = {"source": "https://rexrank.ai (snapshot 2026-10-16), findings track, study level", "test_sets": {}}
    for split, ds in LB_DS.items():
        models = []
        for row in lb[f"testResults{ds}"]:
            vals = {}
            for c, v in zip(LB_COLS, row[2:]):
                try:
                    vals[c] = float(v)
                except (TypeError, ValueError):
                    vals[c] = None
            models.append({"name": row[1], **vals})
        ours = {}
        for run in RUNS:
            s = rex["runs"][run][split]["findings"]["study"]
            ours[run] = {}
            for c in LB_COLS:
                pool = [m[c] for m in models if m[c] is not None]
                ours[run][c] = {"value": s[LB_KEY[c]], "rank": 1 + sum(x > s[LB_KEY[c]] for x in pool),
                                "of": len(pool) + 1}
        out["test_sets"][split] = {"leaderboard": models, "ours": ours}
    return out


def training_curves() -> dict:
    out = {}
    for run in RUNS:
        logs = load(ROOT / "checkpoints" / run / "checkpoint-16072" / "trainer_state.json")["log_history"]
        out[run] = {"eval": [[l["step"], l["eval_loss"]] for l in logs if "eval_loss" in l],
                    "train_final": [l["loss"] for l in logs if "loss" in l][-1]}
    return out


# ------------------------------------------------------------------ figures
def style(ax, th, ygrid=True):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], length=0, labelsize=8.5)
    if ygrid:
        ax.yaxis.grid(True, color=th["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_color(th["ink2"])


def new_fig(th, nrows, ncols, w, h):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    fig.patch.set_facecolor(th["surface"])
    return fig, axes


def legend(fig, th, entries):
    """entries: (label, color) for a bar/dot series, (label, color, "line") for a reference line."""
    handles = [plt.Line2D([], [], color=e[1], linewidth=1.5) if len(e) > 2 else
               plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=e[1]) for e in entries]
    leg = fig.legend(handles, [e[0] for e in entries], loc="upper center",
                     bbox_to_anchor=(0.5, 1 - 0.62 / fig.get_figheight()), ncol=len(entries), frameon=False,
                     fontsize=9, handletextpad=0.5, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(th["ink2"])


def title(fig, th, text, sub=None):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.06 / h, text, ha="left", va="top", fontsize=12, fontweight="bold", color=th["ink"])
    if sub:
        fig.text(0.012, 1 - 0.32 / h, sub, ha="left", va="top", fontsize=8.5, color=th["muted"])


def layout(fig, with_legend=True, left=0.0):
    """tight_layout below the title block (title, subtitle and, optionally, the legend row)."""
    fig.tight_layout(rect=(left, 0, 1, 1 - (0.95 if with_legend else 0.62) / fig.get_figheight()))


def bars(ax, th, groups, series, values, star=None, width=0.3):
    """values[s][g]; star[(s, g)] -> True puts '*' over that bar."""
    x = np.arange(len(groups))
    n = len(series)
    for i, (key, _, color) in enumerate(series):
        off = (i - (n - 1) / 2) * (width + 0.04)
        v = np.array([values[key][g] for g in groups], dtype=float)
        ax.bar(x + off, v, width=width, color=color, linewidth=0, zorder=2)
        for j, g in enumerate(groups):
            if star and star.get((key, g)):
                ax.text(x[j] + off, v[j], "*", ha="center", va="bottom", fontsize=11, color=th["ink2"])
    ax.set_xticks(x)
    ax.axhline(0, color=th["axis"], linewidth=1, zorder=3)


def panel_title(ax, th, text):
    ax.set_title(text, loc="left", fontsize=9.5, color=th["ink"], pad=8)


def save(fig, name, mode):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}_{mode}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_finetuning(idm, mode):
    th = THEMES[mode]
    series = [("medgemma_zeroshot", "MedGemma zero-shot", th["baseline"]),
              ("medgemma_full", RUNS["medgemma_full"], th["medgemma_full"]),
              ("chexone_full", RUNS["chexone_full"], th["chexone_full"])]
    fig, axes = new_fig(th, 1, 3, 11, 3.6)
    for ax, (metric, label) in zip(axes[0], [("bleu_4", "BLEU-4"), ("rouge_l", "ROUGE-L"), ("bertscore_f1", "BERTScore (mBERT)")]):
        vals = {k: {t: idm[k]["test_chexpert"][t][metric] for t in TRACKS} for k, _, _ in series}
        bars(ax, th, list(TRACKS), series, vals, width=0.24)
        ax.set_xticklabels(TRACKS.values())
        style(ax, th)
        panel_title(ax, th, label)
    legend(fig, th, [(n, c) for _, n, c in series])
    title(fig, th, "Fine-tuning teaches the models to write Indonesian reports",
          "CheXpert Plus test set, Indonesian references; zero-shot = MedGemma-1.5-4B with the same Indonesian prompt")
    layout(fig)
    save(fig, "fig1_finetuning_effect", mode)


def fig_indonesian(idm, sig, mode):
    th = THEMES[mode]
    series = [(r, n, th[r]) for r, n in RUNS.items()]
    fig, axes = new_fig(th, 1, 3, 11, 3.6)
    ymax = max(idm[r][s][t]["rouge_l"] for r in RUNS for s in SPLITS for t in TRACKS) * 1.18
    for ax, (split, name) in zip(axes[0], SPLITS.items()):
        vals = {r: {t: idm[r][split][t]["rouge_l"] for t in TRACKS} for r in RUNS}
        star = {}
        for t in TRACKS:
            s = sig["indonesian"].get(f"{split}/{t}", {}).get("rouge_l")
            if s and s["significant"]:
                star[("chexone_full" if s["B_minus_A"] > 0 else "medgemma_full", t)] = True
        bars(ax, th, list(TRACKS), series, vals, star)
        ax.set_xticklabels(TRACKS.values())
        ax.set_ylim(0, ymax)
        style(ax, th)
        panel_title(ax, th, name + (" (external)" if split == "test_openi" else ""))
    legend(fig, th, [(n, c) for _, n, c in series])
    title(fig, th, "Indonesian ROUGE-L by test set",
          "* significant difference (paired bootstrap, 95% CI); F+I was not tested")
    layout(fig)
    save(fig, "fig2_indonesian_rougeL", mode)


def fig_clinical(rex, sig, mode):
    th = THEMES[mode]
    series = [(r, n, th[r]) for r, n in RUNS.items()]
    metrics = [("1/radcliq_v1", "1/RadCliQ-v1", "1/radcliq"), ("semb_score", "SembScore", "semb_score"),
               ("radgraph", "RadGraph", "radgraph"), ("ratescore", "RaTEScore", "ratescore"),
               ("green", "GREEN", "green"), ("crimson", "CRIMSON", "crimson")]
    names = {s: n.replace(" Plus", "+") for s, n in SPLITS.items()}
    fig, axes = new_fig(th, 2, 3, 11, 6.2)
    for ax, (key, label, cmp_key) in zip(axes.flat, metrics):
        vals = {r: {s: rex["runs"][r][s]["findings"]["study"][key] for s in SPLITS} for r in RUNS}
        star = {}
        for s in SPLITS:
            d = sig["rexrank"].get(f"{s}/findings", {}).get(cmp_key)
            if d and d["significant"]:
                star[("chexone_full" if d["B_minus_A"] > 0 else "medgemma_full", s)] = True
        bars(ax, th, list(SPLITS), series, vals, star)
        ax.set_xticklabels([names[s] for s in SPLITS])
        if key in ("semb_score", "radgraph", "ratescore"):
            for j, s in enumerate(SPLITS):
                c = rex["runs"]["reference_backtranslation"][s]["findings"]["study"][key]
                ax.plot([j - 0.36, j + 0.36], [c, c], color=th["ink2"], linewidth=1.2, zorder=4)
            ax.set_ylim(0, 1.0)
        style(ax, th)
        panel_title(ax, th, label)
    legend(fig, th, [(n, c) for _, n, c in series] + [("Translation ceiling (references back-translated)", th["ink2"], "line")])
    title(fig, th, "Clinical metrics through translation (findings, study level)",
          f"Indonesian output translated to English with {rex['translator']} and scored against the original "
          "English reports; GREEN/CRIMSON on 500 studies; * significant (paired bootstrap)")
    layout(fig)
    save(fig, "fig3_clinical_metrics", mode)


def fig_leaderboard(lbd, mode):
    th = THEMES[mode]
    cols = [("1/RadCliQ-v1", "1/RadCliQ-v1"), ("RaTEScore", "RaTEScore")]
    fig, axes = new_fig(th, 3, 2, 11, 5.6)
    rng = np.random.default_rng(0)
    for i, split in enumerate(SPLITS):
        d = lbd["test_sets"][split]
        for j, (c, label) in enumerate(cols):
            ax = axes[i][j]
            v = np.array([m[c] for m in d["leaderboard"] if m[c] is not None])
            ax.scatter(v, rng.uniform(-0.18, 0.18, len(v)), s=26, color=th["baseline"], linewidths=0, zorder=2)
            for k, (run, dy) in enumerate([("medgemma_full", 0.55), ("chexone_full", -0.55)]):
                o = d["ours"][run][c]
                ax.scatter([o["value"]], [0], s=70, color=th[run], edgecolors=th["surface"], linewidths=2, zorder=3)
                ax.annotate(f"{RUNS[run].split('-')[0]} #{o['rank']}", (o["value"], 0), xytext=(0, 14 if dy > 0 else -14),
                            textcoords="offset points", ha="center", va="center", fontsize=8, color=th["ink2"])
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            style(ax, th, ygrid=False)
            ax.xaxis.grid(True, color=th["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(SPLITS[split], rotation=0, ha="right", va="center", fontsize=9, color=th["ink2"])
            if i == 0:
                panel_title(ax, th, label)
    legend(fig, th, [("29 rexrank.ai systems (English)", th["baseline"]),
                     (RUNS["medgemma_full"] + " (Indonesian, via translation)", th["medgemma_full"]),
                     (RUNS["chexone_full"] + " (Indonesian, via translation)", th["chexone_full"])])
    title(fig, th, "Position among the ReXrank leaderboard systems (findings)",
          "Rank out of 30 = the 29 leaderboard systems plus our model; indicative only (our scores pass through translation)")
    layout(fig, left=0.03)
    save(fig, "fig4_leaderboard_position", mode)


def fig_training(curves, mode):
    th = THEMES[mode]
    fig, axes = new_fig(th, 1, 2, 11, 3.4)
    for ax, run in zip(axes[0], RUNS):
        steps, loss = zip(*curves[run]["eval"])
        ax.plot(steps, loss, color=th[run], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(steps[-1:], loss[-1:], s=40, color=th[run], edgecolors=th["surface"], linewidths=2, zorder=3)
        ax.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=th["ink2"])
        ax.set_xlabel("optimizer step", fontsize=8.5, color=th["muted"])
        style(ax, th)
        panel_title(ax, th, f"{RUNS[run]}: development loss")
    title(fig, th, "Training: development loss per checkpoint",
          "Fixed 256-sample CheXpert Plus development subset; not comparable across models (different tokenizers)")
    layout(fig, with_legend=False)
    save(fig, "fig5_training_curves", mode)


# ------------------------------------------------------------------ tables
def fmt(v, nd=3):
    return "–" if v is None or v != v else f"{v:.{nd}f}"


def results_md(idm, rex, sig, lbd, curves) -> str:
    L = ["# Results", "",
         "All numbers are generated by `tools/export_results.py` from the private `outputs/` folder. "
         "Bold = better of the two fine-tuned models; † = significant (paired bootstrap, 95% CI).", ""]
    L += ["## 1. Indonesian metrics (post-processed output vs Indonesian references)", "",
          "| Test set | Track | Model | n | BLEU-1 | BLEU-2 | BLEU-4 | BLEU-2 (ReXrank) | ROUGE-L | METEOR | BERTScore |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    cols = ["bleu_1", "bleu_2", "bleu_4", "bleu2_rexrank", "rouge_l", "meteor", "bertscore_f1"]
    sig_key = {"bleu_4": "bleu_4", "bleu2_rexrank": "bleu2_rexrank", "rouge_l": "rouge_l", "meteor": "meteor"}
    for s, sn in SPLITS.items():
        for t, tn in TRACKS.items():
            rows = [("medgemma_zeroshot", "MedGemma zero-shot")] if s == "test_chexpert" else []
            rows += list(RUNS.items())
            for r, rn in rows:
                m = idm[r][s][t]
                cells = []
                for c in cols:
                    v = m.get(c)
                    txt = fmt(v)
                    if r in RUNS and v is not None:
                        other = idm["chexone_full" if r == "medgemma_full" else "medgemma_full"][s][t].get(c)
                        if other is not None and fmt(v) != fmt(other) and v > other:
                            txt = f"**{txt}**"
                        d = sig["indonesian"].get(f"{s}/{t}", {}).get(sig_key.get(c, ""))
                        if d and d["significant"] and (d["B_minus_A"] > 0) == (r == "chexone_full"):
                            txt += "†"
                    cells.append(txt)
                L.append(f"| {sn} | {tn} | {rn} | {m['n']:,} | " + " | ".join(cells) + " |")
    L += ["", "Zero-shot: unprocessed output, CheXpert Plus only. METEOR uses English synonyms (secondary for Indonesian).", ""]

    L += ["## 2. ReXrank metrics through translation (study level)", "",
          f"Indonesian output translated to English with `{rex['translator']}` and scored against the original English "
          f"reports. GREEN/CRIMSON on {rex['llm_n']} random studies per test set and track. *Ceiling* = Indonesian "
          "references translated back the same way.", "",
          "| Test set | Track | Model | studies | 1/RadCliQ-v1 | BLEU-2 | BERTScore | SembScore | RadGraph | RaTEScore | GREEN | CRIMSON |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rc = [("1/radcliq_v1", "1/radcliq"), ("bleu2", "bleu2"), ("bertscore", "bertscore"), ("semb_score", "semb_score"),
          ("radgraph", "radgraph"), ("ratescore", "ratescore"), ("green", "green"), ("crimson", "crimson")]
    for s, sn in SPLITS.items():
        for t, tn in TRACKS.items():
            for r, rn in list(RUNS.items()) + [("reference_backtranslation", "*Ceiling*")]:
                m = rex["runs"][r][s][t]["study"]
                cells = []
                for key, ck in rc:
                    v = m.get(key)
                    if r == "reference_backtranslation" and key == "1/radcliq_v1":
                        v = None  # RadCliQ is negative for the ceiling: 1/RadCliQ undefined
                    txt = fmt(v)
                    if r in RUNS and v is not None:
                        other = rex["runs"]["chexone_full" if r == "medgemma_full" else "medgemma_full"][s][t]["study"].get(key)
                        if other is not None and fmt(v) != fmt(other) and v > other:
                            txt = f"**{txt}**"
                        d = sig["rexrank"].get(f"{s}/{t}", {}).get(ck)
                        if d and d["significant"] and (d["B_minus_A"] > 0) == (r == "chexone_full"):
                            txt += "†"
                    cells.append(txt)
                L.append(f"| {sn} | {tn} | {rn} | {m['n']:,} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## 3. Rank among the rexrank.ai systems (findings, study level; out of 30)", "",
          "| Test set | Model | 1/RadCliQ | RaTEScore | GREEN | CRIMSON | RadGraph | SembScore | BLEU | BERTScore |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["1/RadCliQ-v1", "RaTEScore", "GREEN", "CRIMSON", "RadGraph", "SembScore", "BLEU", "BertScore"]
    for s, sn in SPLITS.items():
        for r, rn in RUNS.items():
            o = lbd["test_sets"][s]["ours"][r]
            L.append(f"| {sn} | {rn} | " + " | ".join(str(o[c]["rank"]) for c in order) + " |")
    L += ["", f"Source: {lbd['source']}. Indicative only: the leaderboard systems write English reports, ours pass through translation.", ""]

    L += ["## 4. Training", "", "| Model | Steps | Final dev loss | Final train loss |", "|---|---|---|---|"]
    for r, rn in RUNS.items():
        L.append(f"| {rn} | {curves[r]['eval'][-1][0]:,} | {curves[r]['eval'][-1][1]:.4f} | {curves[r]['train_final']:.4f} |")
    L += ["", "Dev loss is not comparable across the two models (different tokenizers).", ""]
    return "\n".join(L)


def main():
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    RES.mkdir(exist_ok=True)
    idm, rex, sig = indonesian_metrics(), rexrank_metrics(), significance()
    lbd, curves = leaderboard(rex), training_curves()
    for obj, name in [(idm, "indonesian_metrics.json"), (rex, "rexrank_metrics.json"), (sig, "significance.json"),
                      (lbd, "leaderboard.json"), (curves, "training_curves.json")]:
        dump(obj, name)
    (RES / "RESULTS.md").write_text(results_md(idm, rex, sig, lbd, curves), encoding="utf-8")
    for mode in THEMES:
        fig_finetuning(idm, mode)
        fig_indonesian(idm, sig, mode)
        fig_clinical(rex, sig, mode)
        fig_leaderboard(lbd, mode)
        fig_training(curves, mode)
    print(f"results written to {RES}")


if __name__ == "__main__":
    main()
