"""
ReXrank-style evaluation (https://rexrank.ai) of rrg_indo predictions.

ReXrank's report-generation columns: 1/RadCliQ-v1 (ranking metric), BLEU(-2),
BertScore, SembScore, RadGraph, RaTEScore, GREEN, CRIMSON - all English-model
metrics. So the Indonesian output is first translated to English (translate.py)
and scored against the ORIGINAL English reference reports of each dataset.

  BLEU-2      per-report BLEU with bigram weights (1/2, 1/2), CXR-Report-Metric tokenization
  BertScore   distilroberta-base, layer 5, baseline-rescaled, idf over the track's references (RadEval)
              bertscore_cxrm: the same without idf - CXR-Report-Metric's default config (USE_IDF = False)
  SembScore   cosine of CheXbert report embeddings
  RadGraph    (entity F1 + relation F1) / 2, original RadGraph model ("radgraph_combined")
  RadCliQ-v1  CXR-Report-Metric's v1 linear model on the four above; reported as 1/mean (higher = better)
              radcliq_v1_cxrm: the same with bertscore_cxrm
  RaTEScore   official RaTEScore
  GREEN       StanfordAIMI/GREEN-radllama2-7b            } LLM judges: fixed random subset of studies
  CRIMSON     rajpurkarlab/medgemma-4b-it-crimson        } (--llm_n per test set per track), cached
Implementations come from RadEval (radeval==2.2.1), whose RadCliQ-v1 is validated
against CXR-Report-Metric, the code ReXrank uses.

Tracks follow ReXrank: findings, findings_impression (impression is extra, non-LLM only).
Scores are reported per image (every test image) and per study (one image per study,
frontal preferred) - ReXrank scores one report per study, so the study level is the
closer comparison. LLM metrics are study level only.

    uv run python rexrank.py --run medgemma_full --csv ../outputs/medgemma_full/predictions/medgemma_full__test_*.csv \
        --out_dir ../outputs/medgemma_full/rexrank --llm green crimson
    # translation-noise ceiling: the Indonesian REFERENCES back-translated, scored against the English originals
    uv run python rexrank.py --run reference_backtranslation --calibration --csv <same CSVs> --out_dir ../outputs/rexrank_calibration
"""

import argparse
import gc
import glob
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _radeval_compat  # noqa: E402,F401 - must precede any radeval import
from rrg.config import CHEXPERT_INDO, MIMIC_INDO, OPENI_INDO  # noqa: E402
from translate import TRANSLATOR, load_cache  # noqa: E402

TRACKS = ["findings", "impression", "findings_impression"]
LLM_TRACKS = ["findings", "findings_impression"]
LLM_CACHE_DIR = HERE.parent / "outputs" / "rexrank_llm_cache"
LLM_SEED = 42
BASE_COLS = ["bleu2", "bertscore", "semb_score", "radgraph", "radcliq_v1", "ratescore"]


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower() if isinstance(text, str) else ""


def _nonempty(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


# ---------------------------------------------------------------------------
# pairs: (English reference, English translation of the Indonesian text)
# ---------------------------------------------------------------------------
def english_references() -> dict:
    """(source, study_id as in the manifest) -> (findings_en, impression_en), original English reports."""
    refs = {}
    cx = pd.read_parquet(CHEXPERT_INDO, columns=["path_to_image", "section_findings", "section_impression"])
    cx["study_id"] = cx["path_to_image"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[0]
    for c in ("section_findings", "section_impression"):
        cx[c] = cx[c].where(cx[c].fillna("").astype(str).str.strip().ne(""))
    for sid, f, i in cx.groupby("study_id")[["section_findings", "section_impression"]].first().itertuples():
        refs[("chexpert_plus", sid)] = (f, i)
    mm = pd.read_csv(MIMIC_INDO["test"], usecols=["subject_id", "study_id", "findings", "impression"])
    for s, st, f, i in mm.itertuples(index=False):
        refs[("mimic_cxr", f"{s}_{st}")] = (f, i)
    oi = pd.read_csv(OPENI_INDO, usecols=["uid", "findings", "impression"])
    for uid, f, i in oi.itertuples(index=False):
        refs[("openi_cxr", uid)] = (f, i)
    return {k: tuple(norm(x) if _nonempty(x) else None for x in v) for k, v in refs.items()}


def build_pairs(df: pd.DataFrame, calibration: bool) -> pd.DataFrame:
    en_refs = english_references()
    trans = load_cache(TRANSLATOR)
    src = "ground_truth" if calibration else "generated"
    missing = Counter()

    def tr(text):
        if not _nonempty(text):
            return ""
        t = trans.get(text.strip())
        if t is None:
            missing["n"] += 1
            return None
        return norm(t)

    rows = []
    for r in df.to_dict("records"):
        ref_f, ref_i = en_refs.get((r["source"], str(r["study_id"])), (None, None))
        hyp_f, hyp_i = tr(r.get(f"{src}_findings")), tr(r.get(f"{src}_impression"))
        has_f = _nonempty(r.get("ground_truth_findings")) and ref_f
        has_i = _nonempty(r.get("ground_truth_impression")) and ref_i
        base = {k: r[k] for k in ("split", "image_path", "study_id", "view")}
        if has_f:
            rows.append({**base, "track": "findings", "ref": ref_f, "hyp": hyp_f})
        if has_i:
            rows.append({**base, "track": "impression", "ref": ref_i, "hyp": hyp_i})
        if has_f and has_i:
            hyp = None if hyp_f is None or hyp_i is None else f"{hyp_f} {hyp_i}".strip()
            rows.append({**base, "track": "findings_impression", "ref": f"{ref_f} {ref_i}", "hyp": hyp})
    if missing["n"]:
        raise SystemExit(f"{missing['n']:,} texts have no translation yet - run translate.py on these CSVs first")
    pairs = pd.DataFrame(rows)
    pairs["hyp"] = pairs["hyp"].where(pairs["hyp"].str.len() > 0, ".")  # empty output scores as '.', not a crash

    # one image per study (frontal first, then path order) - ReXrank scores one report per study
    pairs["_lateral"] = ~pairs["view"].fillna("").astype(str).str.upper().isin(["AP", "PA"])
    order = pairs.sort_values(["split", "track", "study_id", "_lateral", "image_path"])
    rep_idx = order.drop_duplicates(["split", "track", "study_id"]).index
    pairs["study_rep"] = pairs.index.isin(rep_idx)
    return pairs.drop(columns="_lateral")


# ---------------------------------------------------------------------------
# non-LLM metrics
# ---------------------------------------------------------------------------
def _bleu2_tokens(text: str) -> list[str]:
    return [w for w in text.lower().replace(".", " .").split(" ") if w]  # CXR-Report-Metric prep_reports


def bleu2(ref: str, hyp: str) -> float:
    """Sentence BLEU, weights (1/2, 1/2), no smoothing - as fast_bleu in CXR-Report-Metric."""
    r, h = _bleu2_tokens(ref), _bleu2_tokens(hyp)
    if not h:
        return 0.0
    logp = 0.0
    for n in (1, 2):
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        match, total = sum(min(c, rc[g]) for g, c in hc.items()), max(len(h) - n + 1, 0)
        if match == 0 or total == 0:
            return 0.0
        logp += 0.5 * math.log(match / total)
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(logp)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def base_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    from radeval.metrics.RadCliQv1.radcliq import CompositeMetric, compute_f1, extract_entities, extract_relations
    from radeval.metrics.RaTEScore.scorer import RaTEScore
    from radeval.metrics.bertscore._vendor import BERTScorer

    t0 = time.time()
    comp = CompositeMetric()  # RadGraph ('radgraph' model), CheXbert, RadCliQ-v1 scaler + coefficients
    texts = sorted(set(pairs["ref"]) | set(pairs["hyp"]))
    print(f"[rexrank] {len(pairs):,} pairs, {len(texts):,} unique texts", flush=True)

    ents, rels = {}, {}
    for chunk in _chunks(texts, 256):
        out = comp._radgraph(chunk)
        for i, t in enumerate(chunk):
            o = out.get(str(i), {})
            ents[t], rels[t] = extract_entities(o), extract_relations(o)
    print(f"[rexrank] RadGraph done ({time.time() - t0:.0f}s)", flush=True)
    emb = {}
    for chunk in _chunks(texts, 256):
        for t, e in zip(chunk, comp._chexbert.get_embeddings(chunk)):
            emb[t] = np.asarray(e, dtype=np.float64).ravel()
    print(f"[rexrank] CheXbert embeddings done ({time.time() - t0:.0f}s)", flush=True)

    uniq = pairs[["ref", "hyp"]].drop_duplicates().reset_index(drop=True)
    refs, hyps = uniq["ref"].tolist(), uniq["hyp"].tolist()

    def cos(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else 0.0

    uniq["radgraph"] = [(compute_f1(ents[r], ents[h]) + compute_f1(rels[r], rels[h])) / 2 for r, h in zip(refs, hyps)]
    uniq["semb_score"] = [cos(emb[r], emb[h]) for r, h in zip(refs, hyps)]
    uniq["bleu2"] = [bleu2(r, h) for r, h in zip(refs, hyps)]

    # RaTEScore's sentence_split silently drops texts with no sentence of >=3 chars (e.g. empty / "."), which
    # misaligns its outputs. Score those pairs 0.5 up front, the value RaTEScore itself gives when no entities.
    from radeval.metrics.RaTEScore.utils import sentence_split
    ok_text = {t: len(sentence_split([t])[0]) > 0 for t in set(refs) | set(hyps)}
    ok = [ok_text[r] and ok_text[h] for r, h in zip(refs, hyps)]
    print(f"[rexrank] RaTEScore: {ok.count(False)} pair(s) without usable sentences -> 0.5", flush=True)
    rate = RaTEScore(batch_size=16)
    scores = []
    valid = [(r, h) for (r, h), k in zip(zip(refs, hyps), ok) if k]
    for chunk in _chunks(valid, 512):
        s, _, _ = rate.compute_score(candidate_list=[h for _, h in chunk], reference_list=[r for r, _ in chunk])
        assert len(s) == len(chunk), "RaTEScore output misaligned"
        scores.extend(s)
    it = iter(scores)
    uniq["ratescore"] = [next(it) if k else 0.5 for k in ok]
    del rate
    print(f"[rexrank] RaTEScore done ({time.time() - t0:.0f}s)", flush=True)

    pairs = pairs.merge(uniq, on=["ref", "hyp"], how="left")

    # BERTScore idf comes from each (test set, track)'s own references, as one CXR-Report-Metric run would
    pairs["bertscore"] = np.nan
    for (split, track), g in pairs.groupby(["split", "track"]):
        scorer = BERTScorer(model_type="distilroberta-base", num_layers=5, rescale_with_baseline=True, idf=True,
                            idf_sents=g["ref"].tolist(), batch_size=64, nthreads=0, all_layers=False, lang="en")
        u = g[["ref", "hyp"]].drop_duplicates()
        _, _, f1 = scorer.score(cands=u["hyp"].tolist(), refs=u["ref"].tolist(), verbose=False, batch_size=64)
        lookup = dict(zip(zip(u["ref"], u["hyp"]), f1.numpy().tolist()))
        pairs.loc[g.index, "bertscore"] = [lookup[(r, h)] for r, h in zip(g["ref"], g["hyp"])]
        del scorer
    print(f"[rexrank] BERTScore done ({time.time() - t0:.0f}s)", flush=True)

    pairs["radcliq_v1"] = radcliq_v1(comp, pairs, "bertscore")
    pairs = add_cxrm_variant(pairs, comp)

    del comp
    gc.collect()
    torch.cuda.empty_cache()
    return pairs


def radcliq_v1(comp, pairs: pd.DataFrame, bertscore_col: str) -> np.ndarray:
    x = pairs[["radgraph", bertscore_col, "semb_score", "bleu2"]].to_numpy()
    xn = np.hstack([comp.scaler.transform(x), np.ones((len(x), 1))])
    return xn @ comp.coefs  # per-pair RadCliQ-v1 (lower = better)


def add_cxrm_variant(pairs: pd.DataFrame, comp=None) -> pd.DataFrame:
    """BERTScore as CXR-Report-Metric (the code ReXrank cites) computes it with its default config.py,
    USE_IDF = False; 'bertscore' above follows RadEval (idf from the references). Both on the same lowercased
    text as every other metric: CheXpert Plus impressions are all caps, which a translation cannot reproduce,
    and a cased score would measure that. RadCliQ-v1 is recomputed with it, the other three inputs unchanged."""
    from radeval.metrics.bertscore._vendor import BERTScorer
    from radeval.metrics.RadCliQv1.radcliq import CompositeMetric

    t0 = time.time()
    scorer = BERTScorer(model_type="distilroberta-base", num_layers=5, rescale_with_baseline=True, idf=False,
                        batch_size=64, nthreads=0, all_layers=False, lang="en")
    u = pairs[["ref", "hyp"]].drop_duplicates()
    _, _, f1 = scorer.score(cands=u["hyp"].tolist(), refs=u["ref"].tolist(), verbose=False, batch_size=64)
    lookup = dict(zip(zip(u["ref"], u["hyp"]), f1.numpy().tolist()))
    pairs["bertscore_cxrm"] = [lookup[k] for k in zip(pairs["ref"], pairs["hyp"])]
    del scorer
    pairs["radcliq_v1_cxrm"] = radcliq_v1(comp or CompositeMetric(), pairs, "bertscore_cxrm")
    print(f"[rexrank] BERTScore (CXR-Report-Metric default) done ({time.time() - t0:.0f}s)", flush=True)
    return pairs


# ---------------------------------------------------------------------------
# LLM judges (GREEN, CRIMSON): fixed subset, cached per (ref, hyp)
# ---------------------------------------------------------------------------
def llm_subset(pairs: pd.DataFrame, n: int) -> pd.Series:
    """Same studies for every run: depends only on which test studies have a reference, not on predictions."""
    chosen = pd.Series(False, index=pairs.index)
    for (split, track), g in pairs[pairs["study_rep"] & pairs["track"].isin(LLM_TRACKS)].groupby(["split", "track"]):
        g = g.sort_values("study_id")
        rng = np.random.default_rng(LLM_SEED)
        take = rng.choice(len(g), size=min(n, len(g)), replace=False)
        chosen.loc[g.index[np.sort(take)]] = True
    return chosen


def _pair_key(ref: str, hyp: str) -> str:
    return hashlib.sha1(f"{ref}\x00{hyp}".encode("utf-8")).hexdigest()


def _load_llm_cache(metric: str) -> dict:
    p = LLM_CACHE_DIR / f"{metric}.jsonl"
    cache = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d["score"] is not None:  # failed pairs (CRIMSON turns OOM into NaN) are retried next run
                    cache[d["key"]] = d["score"]
    return cache


def _make_llm_scorer(metric: str, batch_size: int):
    if metric == "green":
        from radeval.metrics.green_score.green import GREEN
        scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=str(LLM_CACHE_DIR))
        scorer.batch_size = batch_size
        return scorer
    if metric == "crimson":
        from radeval.metrics.crimson.crimson import CRIMSONScore
        return CRIMSONScore(provider="hf", batch_size=batch_size)
    raise ValueError(metric)


def _score_with_oom_fallback(scorer, metric: str, refs: list, hyps: list) -> list:
    while True:
        try:
            return scorer(refs, hyps)[2]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if scorer.batch_size <= 1:  # a pair too long even alone: record it as failed, keep the rest
                print(f"[rexrank] {metric}: OOM at batch_size 1 -> {len(refs)} pair(s) left unscored", flush=True)
                return [None] * len(refs)
            scorer.batch_size = max(1, scorer.batch_size // 2)
            print(f"[rexrank] {metric}: OOM -> batch_size {scorer.batch_size}", flush=True)


def llm_metric(pairs: pd.DataFrame, metric: str, chunk: int, batch_size: int, cache_only: bool = False) -> pd.Series:
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_llm_cache(metric)
    sub = pairs[pairs["llm_subset"]]
    todo = {}
    for r, h in zip(sub["ref"], sub["hyp"]):
        k = _pair_key(r, h)
        if k not in cache:
            todo[k] = (r, h)
    print(f"[rexrank] {metric}: {len(sub):,} subset pairs, {len(todo):,} not cached"
          + (" (cache only: left out of the mean)" if cache_only and todo else ""), flush=True)
    if todo and not cache_only:
        scorer = _make_llm_scorer(metric, batch_size)
        items, t0, failed = list(todo.items()), time.time(), 0
        with open(LLM_CACHE_DIR / f"{metric}.jsonl", "a", encoding="utf-8") as f:
            for i, part in enumerate(_chunks(items, chunk)):
                scores = _score_with_oom_fallback(scorer, metric, [r for _, (r, _) in part], [h for _, (_, h) in part])
                for (k, (r, h)), s in zip(part, scores):
                    s = None if s is None or (isinstance(s, float) and math.isnan(s)) else float(s)
                    cache[k] = s
                    failed += s is None
                    f.write(json.dumps({"key": k, "score": s, "ref": r, "hyp": h}) + "\n")
                f.flush()
                done = min((i + 1) * chunk, len(items))
                rate = done / (time.time() - t0)
                print(f"[rexrank] {metric}: {done:,}/{len(items):,} ({rate * 60:.1f} pairs/min, "
                      f"~{(len(items) - done) / rate / 3600:.1f} h left, {failed} failed)", flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.Series(np.nan, index=pairs.index)
    # unscored pairs (failed or not cached) -> NaN, left out of the mean; a None would make pandas reject the column
    vals = [cache.get(_pair_key(r, h)) for r, h in zip(sub["ref"], sub["hyp"])]
    out.loc[sub.index] = [np.nan if v is None else float(v) for v in vals]
    return out.astype(float)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def summarize(pairs: pd.DataFrame, llm: list[str]) -> dict:
    res = {}
    for (split, track), g in pairs.groupby(["split", "track"]):
        res.setdefault(split, {})[track] = {}
        for level, lg in (("image", g), ("study", g[g["study_rep"]])):
            m = {"n": int(len(lg)),
                 "1/radcliq_v1": float(1 / lg["radcliq_v1"].mean()),
                 "radcliq_v1": float(lg["radcliq_v1"].mean()),  # 1/x is meaningless once the mean nears or drops below 0
                 "bleu2": float(lg["bleu2"].mean()),
                 "bertscore": float(lg["bertscore"].mean()),
                 "semb_score": float(lg["semb_score"].mean()),
                 "radgraph": float(lg["radgraph"].mean()),
                 "ratescore": float(lg["ratescore"].mean())}
            if "bertscore_cxrm" in lg:
                m["bertscore_cxrm"] = float(lg["bertscore_cxrm"].mean())
                m["1/radcliq_v1_cxrm"] = float(1 / lg["radcliq_v1_cxrm"].mean())
                m["radcliq_v1_cxrm"] = float(lg["radcliq_v1_cxrm"].mean())
            if level == "study":
                for metric in llm:
                    v = lg[metric].dropna()
                    if len(v):
                        m[metric] = float(v.mean())
                        m[f"{metric}_n"] = int(len(v))
            res[split][track][level] = m
    return res


def markdown_table(run: str, res: dict) -> str:
    cols = ["1/radcliq_v1", "radcliq_v1", "bleu2", "bertscore", "semb_score", "radgraph", "ratescore", "green", "crimson",
            "bertscore_cxrm", "1/radcliq_v1_cxrm"]
    lines = [f"# ReXrank-style metrics - {run}", "",
             "English translation (MedGemma-1.5-4b-it) of the Indonesian output vs. the original English reports. "
             "Study level = one image per study (frontal first); GREEN/CRIMSON on a fixed random subset of studies. "
             "`_cxrm`: BERTScore as CXR-Report-Metric's default config computes it (no idf), and "
             "RadCliQ-v1 with that BERTScore.", "",
             "| test set | track | level | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (4 + len(cols))]
    for split, tracks in res.items():
        for track, levels in tracks.items():
            for level, m in levels.items():
                vals = [f"{m[c]:.3f}" + (f" (n={m[c + '_n']})" if c + "_n" in m else "") if c in m else "-" for c in cols]
                lines.append(f"| {split} | {track} | {level} | {m['n']} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="name used in output files, e.g. medgemma_full")
    p.add_argument("--csv", nargs="+", required=True, help="prediction CSVs (globs ok), one or more test sets")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--calibration", action="store_true",
                   help="score the back-translated Indonesian REFERENCES instead of the model output")
    p.add_argument("--llm", nargs="*", default=[], choices=["green", "crimson"])
    p.add_argument("--llm_n", type=int, default=500, help="studies per test set per track for GREEN/CRIMSON")
    p.add_argument("--llm_chunk", type=int, default=20, help="pairs scored between cache writes")
    p.add_argument("--green_batch_size", type=int, default=4)
    p.add_argument("--crimson_batch_size", type=int, default=1)
    p.add_argument("--llm_cache_only", action="store_true",
                   help="no GPU: summarize GREEN/CRIMSON from the cache; unscored pairs are left out (n is reported)")
    p.add_argument("--limit", type=int, default=None, help="first N rows per CSV (smoke test)")
    p.add_argument("--reuse", action="store_true",
                   help="load base metrics from an existing per-sample parquet (e.g. to add --llm afterwards)")
    args = p.parse_args()

    csvs = sorted({f for pat in args.csv for f in glob.glob(pat)})
    if not csvs:
        raise SystemExit(f"no CSV matched {args.csv}")
    df = pd.concat([pd.read_csv(c).head(args.limit) if args.limit else pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f"[rexrank] {args.run}: {len(df):,} rows from {len(csvs)} CSV(s)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / f"{args.run}__rexrank_per_sample.parquet"

    if args.reuse and per_sample_path.exists():
        pairs = pd.read_parquet(per_sample_path)
        print(f"[rexrank] reusing base metrics from {per_sample_path}", flush=True)
        if "bertscore_cxrm" not in pairs:  # written before the CXR-Report-Metric variant was added
            pairs = add_cxrm_variant(pairs)
    else:
        pairs = base_metrics(build_pairs(df, args.calibration))
    pairs["llm_subset"] = llm_subset(pairs, args.llm_n)
    pairs.to_parquet(per_sample_path, index=False)  # base metrics are safe on disk before the slow part

    for metric in args.llm:
        bs = args.green_batch_size if metric == "green" else args.crimson_batch_size
        try:
            pairs[metric] = llm_metric(pairs, metric, args.llm_chunk, bs, args.llm_cache_only)
        except Exception as e:  # one judge failing (OOM, download) must not lose the rest
            print(f"[rexrank] {metric} FAILED: {type(e).__name__}: {e}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        pairs.to_parquet(per_sample_path, index=False)

    llm_done = [m for m in args.llm if m in pairs]
    res = summarize(pairs, llm_done)
    meta = {"run": args.run, "calibration": args.calibration, "translator": TRANSLATOR, "csvs": csvs,
            "llm_n": args.llm_n, "llm_seed": LLM_SEED, "results": res}
    (out_dir / f"{args.run}__rexrank.json").write_text(json.dumps(meta, indent=2))
    md = markdown_table(args.run, res)
    (out_dir / f"{args.run}__rexrank.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[rexrank] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
