#!/usr/bin/env bash
# Run a benchmark command only while the GPU is otherwise idle.
#
#   bench/quiet.sh .venv/bin/python bench/bench.py --quick --label foo
#
# Timing numbers are meaningless if another GPU job runs at the same time
# (another benchmark, a training/inference job, a test suite).  This wrapper
#   1. waits until no process matching $BUSY_PATTERN is alive,
#   2. runs the command while watching for such a process in the background,
#   3. if one appeared mid-run, deletes the newest result file and retries.
# Edit BUSY_PATTERN for whatever else uses your GPU.
BUSY_PATTERN=${BUSY_PATTERN:-"demucs|pytest|bench/(bench|noise|peak_bw)\.py"}
RESULTS=bench/results
busy() { pgrep -f -E "$BUSY_PATTERN" | grep -v -E "^($$|$1)$" >/dev/null; }

for attempt in 1 2 3 4 5 6; do
  while busy; do echo "[quiet] GPU busy, waiting..."; sleep 20; done
  before=$(ls -t $RESULTS/*.json 2>/dev/null | head -1)
  "$@" & pid=$!
  dirty=0
  while kill -0 $pid 2>/dev/null; do
    # ignore our own benchmark process (and its children)
    if pgrep -f -E "$BUSY_PATTERN" | grep -v -E "^($$|$pid)$" | while read p; do
         [ "$(ps -o ppid= -p $p | tr -d ' ')" = "$pid" ] || echo $p; done | grep -q .; then
      dirty=1
    fi
    sleep 1
  done
  wait $pid; rc=$?
  if [ $dirty = 0 ]; then exit $rc; fi
  after=$(ls -t $RESULTS/*.json 2>/dev/null | head -1)
  if [ -n "$after" ] && [ "$after" != "$before" ]; then
    echo "[quiet] another GPU job ran during the measurement; discarding $after"; rm -f "$after"
  fi
done
echo "[quiet] gave up after 6 attempts"; exit 1
