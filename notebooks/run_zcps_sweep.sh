#!/bin/bash
# Sweep zcps benchmarks over version x scheme x order x resolution.
PY=~/venvs/spd-gpu/bin/python
BENCH=~/spd/notebooks/bench_zcps.py
OUT=~/spd/notebooks/zcps_results.jsonl
: > "$OUT"

for repo in ~/spd ~/spd-baseline; do
  for sim in sd sdfb; do
    for p in 3 7; do
      for R in 32 64 128 256; do
        case $R in
          32|64) steps=20 ;;
          128)   steps=10 ;;
          256)   steps=4 ;;
        esac
        echo ">>> repo=$repo sim=$sim p=$p R=$R steps=$steps"
        SPD_REPO=$repo $PY "$BENCH" --p $p --R $R --sim $sim --steps $steps 2>&1 \
          | grep '^RESULT' | sed 's/^RESULT //' >> "$OUT" \
          || echo "{\"repo\": \"$repo\", \"sim\": \"$sim\", \"p\": $p, \"R\": $R, \"failed\": true}" >> "$OUT"
      done
    done
  done
done
echo "sweep done"
