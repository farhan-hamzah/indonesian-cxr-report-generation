"""
Text normalization.

  clean_text    applied by prepare_data.py to every report section (training targets AND test
                references), so both sides are normalized the same way
  clean_target  extra cleanup for training targets only (train.py); test references stay as in
                the manifest so metrics remain comparable across runs and with the English baseline
"""

import re

# ---------------------------------------------------------------------------
# clean_text (adopted from ../repo/train_stage_a.py)
# ---------------------------------------------------------------------------
MEDICAL_DICT = {
    "kekeruhan": "opasitas",
    "wilayah udara": "ruang udara",
    "penumpulan": "blunting",
    "sudut costofrenum": "sudut kostofrenikus",
    "sudut kosto": "sudut kostofrenikus",
    "tabung nasogastrik": "selang nasogastrik",
    "tabung endotrakeal": "selang endotrakeal",
    "kabel alat pacu jantung": "kawat alat pacu jantung",
    "tidak ada kelainan akut": "tidak tampak kelainan akut",
    "tidak ada temuan akut": "tidak tampak kelainan akut",
}
_DICT_RE = [(re.compile(r"\b" + re.escape(k) + r"\b"), v) for k, v in MEDICAL_DICT.items()]

NON_SUBSTANTIAL = re.compile(
    r"^\s*(|/|-|\.|dan|and|none|tidak ada|lihat kesan\.?|see impression\.?)\s*$", re.IGNORECASE)
SECTION_PREFIX = re.compile(r"^\s*(temuan|kesan)\s*:\s*", re.IGNORECASE)


def clean_text(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = SECTION_PREFIX.sub("", text.strip()).lower()
    for pattern, repl in _DICT_RE:
        t = pattern.sub(repl, t)
    t = re.sub(r"\s+", " ", t).strip()
    return None if NON_SUBSTANTIAL.match(t) else t


# ---------------------------------------------------------------------------
# clean_target (training targets only)
# ---------------------------------------------------------------------------
# sentences that are report logistics, not image content: consult phone lines,
# attending attestations, "findings communicated to Dr. X by phone at 15.30"
JUNK_SENTENCE = re.compile(
    r"konsultasi radiologi|\(\d{3}\)|\d{3}[- ]\d{4}|saya telah meninjau|saya sudah meninjau"
    r"|dikomunikasikan|didiskusikan dengan|diberitahukan|melalui telepon|via telepon")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
ORPHAN_NUMBER = re.compile(r"(?:^|\s)\d+\.\s*(?=\d+\.\s|$)")
# the consult-line phrase often trails a finding without a full stop, so cut it inline
CONSULT_PHONE = re.compile(
    r'[("\s]*(?:(?:layanan|saluran|garis|jalur|nomor)\s+)?konsultasi radiologi[^:]{0,40}:[\s\d()\-]+["\)]*')
ANON_BLANK = re.compile(r"_{2,}")  # MIMIC de-identification blanks ("dibandingkan dengan ___")
# translation failures that loop on a 1-3 word phrase ("kemudian, kemudian, kemudian, ...")
DEGENERATE = re.compile(r"((?:\S+\s+){1,3}?)\1{4,}")


def clean_target(text) -> str | None:
    """Extra cleanup for training targets on top of clean_text; None drops the sample."""
    if not isinstance(text, str) or DEGENERATE.search(text + " "):
        return None
    t = CONSULT_PHONE.sub(" ", text)
    t = ANON_BLANK.sub("sebelumnya", t)
    t = re.sub(r"\bsebelumnya(\s+sebelumnya)+\b", "sebelumnya", t)
    t = " ".join(s for s in SENTENCE_SPLIT.split(t) if not JUNK_SENTENCE.search(s))
    t = ORPHAN_NUMBER.sub(" ", t).replace('"', "")
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t.split()) >= 2 else None
