#!/bin/bash
# 2D zcps sweep: version x scheme x order x resolution.
PY=~/venvs/spd-gpu/bin/python
BENCH=~/spd/notebooks/bench_zcps.py
OUT=~/spd/notebooks/zcps_results_2d.jsonl
: > "$OUT"

for repo in ~/spd ~/spd-baseline; do
  for sim in sd sdfb; do
    for p in 3 7; do
      for R in 64 128 256 512 1024 2048; do
        steps=20; [ $R -ge 2048 ] && steps=10
        echo ">>> repo=$repo sim=$sim p=$p R=$R steps=$steps"
        SPD_REPO=$repo $PY "$BENCH" --p $p --R $R --ndim 2 --sim $sim --steps $steps 2>&1 \
          | grep '^RESULT' | sed 's/^RESULT //' >> "$OUT" \
          || echo "{\"repo\": \"$repo\", \"sim\": \"$sim\", \"p\": $p, \"R\": $R, \"ndim\": 2, \"failed\": true}" >> "$OUT"
      done
    done
  done
done
echo "sweep done"
