# rexrank_eval — ReXrank-style metrics for the Indonesian model

Separate uv environment (RadEval 2.2.1 + torch 2.11 cu128), so the training venv in
`rrg_indo/.venv` is never touched.

ReXrank (https://rexrank.ai) scores report generation with 1/RadCliQ-v1 (ranking metric),
BLEU-2, BertScore, SembScore, RadGraph, RaTEScore, GREEN and CRIMSON. Except BLEU, all of
them run English-only models (CheXbert, RadGraph, LLM judges). So:

1. `translate.py` translates the Indonesian output to English with `google/medgemma-1.5-4b-it`
   (the base model, not our adapter), cached in `../outputs/translation_cache/`.
2. `rexrank.py` scores the translation against the **original English** reference reports
   (CheXpert-Plus `section_findings/impression`, MIMIC `findings/impression`, OpenI `findings/impression`).
3. `--calibration` scores the back-translated **Indonesian references** the same way. That is the
   ceiling translation noise allows; read model scores relative to it.

Indonesian-side metrics (BLEU-1..4, ROUGE-L, METEOR, BERTScore on Indonesian text) stay in
`../evaluate.py` and are reported alongside.

```powershell
cd rexrank_eval                 # from the repository root
uv sync --python 3.12
uv run python translate.py --csv ..\outputs\medgemma_full\predictions\medgemma_full__test_mimic.csv
uv run python rexrank.py --run medgemma_full --csv "..\outputs\medgemma_full\predictions\medgemma_full__test_*.csv" --out_dir ..\outputs\medgemma_full\rexrank
# LLM judges afterwards, reusing the base metrics:
uv run python rexrank.py --run medgemma_full --csv "..\outputs\medgemma_full\predictions\medgemma_full__test_*.csv" --out_dir ..\outputs\medgemma_full\rexrank --reuse --llm green crimson
```

Outputs per run: `<run>__rexrank.md` (table), `<run>__rexrank.json`, `<run>__rexrank_per_sample.parquet`.

## Comparability with the leaderboard

- Tracks: `findings` and `findings_impression` as on ReXrank (`impression` is extra).
- `study` level = one image per study (frontal first), since ReXrank scores one report per study;
  `image` level = every test image.
- Test sets differ from ReXrank's: MIMIC official test (ours: all studies with a translated
  reference; ReXrank: 2,347 studies), CheXpert-Plus valid (same split), OpenI all 3,826 studies
  (ReXrank: the 590-study R2Gen test split). Treat leaderboard comparisons as indicative.
- GREEN / CRIMSON run on a fixed random subset (`--llm_n` studies per test set per track, seed 42,
  the same studies for every run). Scores are cached per (reference, hypothesis) pair in
  `../outputs/rexrank_llm_cache/`, so interrupted runs resume.
- The table also lists mean `radcliq_v1` (lower = better): per-pair RadCliQ-v1 goes negative for near-perfect
  pairs (identical text: -1.44), so 1/mean only reads sensibly while the mean stays clearly above 0.
- Failed LLM-judge pairs (CRIMSON turns OOM/parse errors into NaN) are not cached and are retried on the next run.
- `_radeval_compat.py` works around radeval importing itself as `RadEval` (fails on Windows).
