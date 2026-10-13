#!/bin/bash
# Lightweight live status monitor. Uses macOS command-line tools, not Python.

RUN_NAME="${1:-transfer_full}"
INTERVAL="${2:-30}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORTS="$PROJECT_DIR/outputs/reports"
CHECKPOINT="$PROJECT_DIR/outputs/checkpoints/$RUN_NAME.pt"
HISTORY="$REPORTS/${RUN_NAME}_history.json"
SUMMARY="$REPORTS/${RUN_NAME}_training.json"

case "$INTERVAL" in
  ''|*[!0-9]*) echo "Interval must be whole seconds (default 30)." >&2; exit 2 ;;
esac
if [ "$INTERVAL" -lt 5 ]; then
  echo "Interval must be at least 5 seconds." >&2
  exit 2
fi

echo "Live status for $RUN_NAME; updating every ${INTERVAL}s. Press Ctrl+C to stop (training continues)."

while true; do
  NOW="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  EPOCHS=0
  if [ -f "$HISTORY" ]; then
    EPOCHS="$(grep -c '"epoch"[[:space:]]*:' "$HISTORY" 2>/dev/null || true)"
    EPOCHS="${EPOCHS:-0}"
  fi

  PROCESS="$(ps -Ao pid=,etime=,%cpu=,command= 2>/dev/null | awk -v run="$RUN_NAME" '
    index($0, "-m sanskrit_ocr") && index($0, "train --model transfer") && index($0, "--run-name " run) && !index($0, "awk -v run=") && !index($0, "ps -Ao") { print; exit }
  ')"

  if [ -n "$PROCESS" ]; then
    set -- $PROCESS
    STATUS="TRAINING ACTIVE | PID $1 | elapsed $2 | CPU $3% | last completed epoch $EPOCHS"
  elif [ -f "$SUMMARY" ]; then
    DONE_EPOCHS="$(grep -m1 '"epochs_completed"' "$SUMMARY" | sed 's/[^0-9]//g')"
    BEST="$(grep -m1 '"best_validation_accuracy"' "$SUMMARY" | sed 's/.*: *//; s/[, ].*//')"
    STATUS="TRAINING COMPLETE | ${DONE_EPOCHS:-$EPOCHS} epochs | best validation accuracy ${BEST:-see report}"
  else
    STATUS="TRAINING PROCESS NOT FOUND | last completed epoch $EPOCHS"
  fi

  if [ -f "$CHECKPOINT" ]; then
    MTIME="$(stat -f '%m' "$CHECKPOINT" 2>/dev/null || true)"
    if [ -n "$MTIME" ]; then
      AGE=$(( $(date +%s) - MTIME ))
      STATUS="$STATUS | checkpoint saved $((AGE / 60))m $((AGE % 60))s ago"
    else
      STATUS="$STATUS | checkpoint exists"
    fi
  else
    STATUS="$STATUS | no checkpoint saved yet"
  fi

  echo "[$NOW] $STATUS"
  sleep "$INTERVAL"
done
