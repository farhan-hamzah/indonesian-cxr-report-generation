#!/bin/bash
# Full evaluation of one trained run, any base model (Git Bash):
#   bash scripts/eval_run.sh <run_name> [from_step] [wait_pid]
#
#   1. infer      test_chexpert / test_mimic / test_openi with checkpoints/<run>/final_adapter
#                 -> outputs/<run>/predictions (post-processed + raw columns),
#                 Indonesian metrics -> outputs/<run>/metrics
#   2. translate  ID->EN translation of predictions + references (rexrank_eval/translate.py, cached)
#   3. rexrank    ReXrank metrics without LLM judges -> outputs/<run>/rexrank
#   4. llm        GREEN + CRIMSON on the fixed study subset (slowest step; cached, resumable)
#
# from_step = infer (default) | translate | rexrank | llm: start there (earlier results are reused).
# wait_pid  = wait for that process (e.g. the training run) to exit first; 0 = don't wait.
# Every step is safe to re-run. Progress: outputs/<run>/eval_run.log
#
#   bash scripts/eval_run.sh chexone_full
#   bash scripts/eval_run.sh medgemma_full rexrank

RUN=${1:?usage: bash scripts/eval_run.sh <run_name> [infer|translate|rexrank|llm] [wait_pid]}
FROM=${2:-infer}
WAIT_PID=${3:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT=outputs/$RUN
ADAPTER=checkpoints/$RUN/final_adapter
SPLITS="test_chexpert test_mimic test_openi"
stage_no() { case $1 in infer) echo 1;; translate) echo 2;; rexrank) echo 3;; llm) echo 4;; *) echo 0;; esac; }
[ "$(stage_no "$FROM")" = 0 ] && { echo "from_step must be infer, translate, rexrank or llm"; exit 1; }
from() { [ "$(stage_no "$1")" -ge "$(stage_no "$FROM")" ]; }  # true if this step is at or after FROM
if [ ! -d "checkpoints/$RUN" ] && [ ! -d "$OUT/predictions" ]; then
  echo "no checkpoints/$RUN or $OUT/predictions - check the run name"; exit 1
fi
mkdir -p $OUT/predictions $OUT/metrics
LOG="$ROOT/$OUT/eval_run.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ "$WAIT_PID" != "0" ]; then
  log "waiting for PID $WAIT_PID"
  while powershell -NoProfile -c "if (Get-Process -Id $WAIT_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do
    sleep 300
  done
  log "PID $WAIT_PID exited"
  sleep 30  # let GPU memory release
fi
if from infer && [ ! -f "$ADAPTER/run_args.json" ]; then
  log "ERROR: $ADAPTER/run_args.json missing -> training did not finish cleanly; stopping"
  exit 2
fi
log "eval $RUN from step: $FROM"

# 1. inference + Indonesian metrics
if from infer; then
  for S in $SPLITS; do
    CSV=$OUT/predictions/${RUN}__$S.csv
    log "infer $S"
    uv run python infer.py --adapter $ADAPTER --split $S --output_csv $CSV --resume >> "$LOG" 2>&1 \
      || { log "infer $S failed, retry with batch_size 4";
           uv run python infer.py --adapter $ADAPTER --split $S --output_csv $CSV --resume --batch_size 4 >> "$LOG" 2>&1 \
           || { log "ERROR infer $S failed twice"; continue; }; }
    log "Indonesian metrics $S (post-processed + raw)"
    uv run python evaluate.py --predictions_csv $CSV --metric_dir $OUT/metrics >> "$LOG" 2>&1 || log "ERROR evaluate $S"
    uv run python evaluate.py --predictions_csv $CSV --metric_dir $OUT/metrics --raw >> "$LOG" 2>&1 || log "ERROR evaluate --raw $S"
  done
fi

# 2-4. ReXrank (English) metrics, in rexrank_eval's own environment
cd rexrank_eval || exit 1
PRED="../$OUT/predictions/${RUN}__test_*.csv"
if from translate; then
  log "translate ID->EN"
  uv run python translate.py --csv $PRED >> "$LOG" 2>&1 || { log "ERROR translate; stopping"; exit 3; }
fi
if from rexrank; then
  log "rexrank $RUN (non-LLM metrics)"
  uv run python rexrank.py --run "$RUN" --csv "$PRED" --out_dir ../$OUT/rexrank >> "$LOG" 2>&1 \
    || { log "ERROR rexrank $RUN; stopping"; exit 4; }
fi
log "rexrank $RUN (GREEN + CRIMSON)"
uv run python rexrank.py --run "$RUN" --csv "$PRED" --out_dir ../$OUT/rexrank --reuse --llm green crimson >> "$LOG" 2>&1 \
  || { log "ERROR rexrank --llm $RUN"; exit 5; }
log "DONE $RUN -> $OUT/{metrics,rexrank}"
