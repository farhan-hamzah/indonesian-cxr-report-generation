"""
Post-processing for generated reports: removes within-report repetition that
greedy decoding sometimes produces. Applied to model output only - references
are never touched.

  1. a phrase looping back-to-back 3+ times ("kemudian, kemudian, kemudian, ...") -> once
  2. a sentence / numbered item that repeats an earlier one (case, punctuation and
     list numbers ignored) -> dropped, numbered lists renumbered
  3. a dangling unfinished last sentence (generation hit max_new_tokens) -> dropped

Near-duplicates that differ in content ("... paru kanan ..." vs "... paru kiri ...")
are kept on purpose: they are separate findings, not repetition.

infer.py applies this automatically (generated_<task> = cleaned, generated_<task>_raw =
untouched). tools/postprocess_csv.py adds the columns to older prediction CSVs.
"""

import re

PHRASE_LOOP = re.compile(r"((?:\S+\s+){1,8}?)\1{2,}")
DANGLING_NUMBER = re.compile(r"\s+\d{1,2}\.?\s*$")  # "... kanan. 13" left by a cut-off list
NUMBERED = re.compile(r"^\s*1\.\s+")
ITEM_MARKER = re.compile(r"(?:^|\s+)\d{1,2}\.\s+(?=\S)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _key(unit: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unit.lower())).strip()


def _dedupe(units: list[str]) -> list[str]:
    seen, kept = set(), []
    for u in units:
        k = _key(u)
        if k and k not in seen:
            seen.add(k)
            kept.append(u)
    return kept


def _drop_unfinished_tail(units: list[str]) -> list[str]:
    if len(units) > 1 and not units[-1].rstrip().endswith((".", "!", "?")):
        return units[:-1]
    return units


def postprocess(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    t = PHRASE_LOOP.sub(r"\1", text.strip() + " ").strip()
    t = DANGLING_NUMBER.sub("", t)

    if NUMBERED.match(t):
        items = [s.strip() for s in ITEM_MARKER.split(t) if s.strip()]
        items = [" ".join(_dedupe(SENTENCE_END.split(it))) for it in items]
        items = _drop_unfinished_tail(_dedupe(items))
        out = " ".join(f"{i}. {it}" for i, it in enumerate(items, 1))
    else:
        sentences = [s.strip() for s in SENTENCE_END.split(t) if s.strip()]
        out = " ".join(_drop_unfinished_tail(_dedupe(sentences)))
    return out or text.strip()
