#!/bin/bash

set -e
set -o pipefail

ROOT_DIR="Data-set"
PROFILE_DIR="outputs/profiles"
REPORT_DIR="outputs/reports"
LOG_DIR="outputs/logs"

N_SPLITS=5
N_REPEATS=3
RANDOM_STATE=42

mkdir -p "$PROFILE_DIR"
mkdir -p "$REPORT_DIR"
mkdir -p "$LOG_DIR"

echo "Starting DAQUA full pipeline..."
echo "Dataset root: $ROOT_DIR"
echo "Profile output: $PROFILE_DIR"
echo "Report output: $REPORT_DIR"
echo "Log output: $LOG_DIR"
echo "CV splits: $N_SPLITS"
echo "CV repeats: $N_REPEATS"
echo ""

python3 -m daqua.main \
  --root "$ROOT_DIR" \
  --out-dir "$PROFILE_DIR" \
  --n-splits "$N_SPLITS" \
  --n-repeats "$N_REPEATS" \
  --random-state "$RANDOM_STATE" \
  2>&1 | tee "$LOG_DIR/DAQUA_pipeline.log"

echo ""
echo "Generating DAQUA summary report..."

python3 -m daqua.reports.report_generator \
  --profiles-dir "$PROFILE_DIR" \
  --out "$REPORT_DIR/DAQUA_summary_report.md" \
  2>&1 | tee "$LOG_DIR/DAQUA_report.log"

echo ""
echo "DAQUA full pipeline completed."
echo "Main readiness file: $PROFILE_DIR/DAQUA_readiness_scores.csv"
echo "Summary report: $REPORT_DIR/DAQUA_summary_report.md"
echo "Pipeline log: $LOG_DIR/DAQUA_pipeline.log"
echo "Report log: $LOG_DIR/DAQUA_report.log"