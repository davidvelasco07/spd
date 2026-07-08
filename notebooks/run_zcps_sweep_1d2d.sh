#!/bin/bash
# Sweep zcps benchmarks in 1D and 2D over version x scheme x order x resolution.
PY=~/venvs/spd-gpu/bin/python
BENCH=~/spd/notebooks/bench_zcps.py
OUT=~/spd/notebooks/zcps_results_1d2d.jsonl
: > "$OUT"

run () {
  local repo=$1 sim=$2 p=$3 R=$4 ndim=$5 steps=$6
  echo ">>> repo=$repo ndim=${ndim}D sim=$sim p=$p R=$R steps=$steps"
  SPD_REPO=$repo $PY "$BENCH" --p $p --R $R --ndim $ndim --sim $sim --steps $steps 2>&1 \
    | grep '^RESULT' | sed 's/^RESULT //' >> "$OUT" \
    || echo "{\"repo\": \"$repo\", \"sim\": \"$sim\", \"p\": $p, \"R\": $R, \"ndim\": $ndim, \"failed\": true}" >> "$OUT"
}

for repo in ~/spd ~/spd-baseline; do
  for sim in sd sdfb; do
    for p in 3 7; do
      # ---- 2D ----
      for R in 64 128 256 512 1024 2048; do
        steps=20; [ $R -ge 2048 ] && steps=10
        run $repo $sim $p $R 2 $steps
      done
      # ---- 1D ----
      for R in 4096 16384 65536 262144 1048576; do
        run $repo $sim $p $R 1 20
      done
    done
  done
done
echo "sweep done"
