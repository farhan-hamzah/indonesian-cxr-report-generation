"""
Indonesian NLG metrics, all on ReXrank's 0-1 scale.

  bleu_1..4      corpus BLEU, same sacrebleu settings as ../rrg_pipeline/evaluate.py, divided by 100
                 (divide the English run's numbers by 100 to compare)
  bleu2_rexrank  ReXrank's own BLEU column: mean per-report BLEU-2, weights (1/2, 1/2), no smoothing,
                 CXR-Report-Metric tokenization (same as rexrank_eval/rexrank.py, here on Indonesian text)
  rouge_l        mean per-report ROUGE-L F1, no stemming (Porter stemmer is English-only)
  meteor         English WordNet synonyms, so for Indonesian it reduces to exact matches - secondary
  bertscore_f1   multilingual BERT (lang="id")

English-only clinical metrics (CheXbert, RadGraph, RaTEScore, GREEN, ...) are computed on
translations in rexrank_eval/.
"""

import math
from collections import Counter

import numpy as np


def _bleu2_tokens(text: str) -> list[str]:
    return [w for w in text.lower().replace(".", " .").split(" ") if w]  # CXR-Report-Metric prep_reports


def bleu2_rexrank(ref: str, hyp: str) -> float:
    """Sentence BLEU, weights (1/2, 1/2), no smoothing - as fast_bleu in CXR-Report-Metric (ReXrank)."""
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


def compute_nlg_metrics(preds: list[str], refs: list[str], bertscore: bool = True) -> dict:
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer

    preds = [p if p.strip() else "." for p in preds]  # empty strings crash bert-score / meteor
    refs = [r if r.strip() else "." for r in refs]

    out = {f"bleu_{n}": BLEU(max_ngram_order=n, effective_order=True, smooth_method="exp")
           .corpus_score(preds, [refs]).score / 100 for n in (1, 2, 3, 4)}
    out["bleu2_rexrank"] = float(np.mean([bleu2_rexrank(r, p) for r, p in zip(refs, preds)]))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    out["rouge_l"] = float(np.mean([rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]))

    import nltk
    from nltk.translate.meteor_score import meteor_score
    for resource in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[1], quiet=True)
    out["meteor"] = float(np.mean([meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]))

    if bertscore:
        from bert_score import score
        _, _, f1 = score(preds, refs, lang="id", verbose=False)
        out["bertscore_f1"] = float(f1.mean())
    return out
