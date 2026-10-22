#!/usr/bin/env bash
# Reproduce the reported results from scratch (assumes the dataset is in data/circor).
#
#   bash scripts/run_all.sh
#
# Times on an Apple M-series laptop: ~45 min per CNN CV run, ~60 min per RNN CV run.
# The CNN uses the GPU (MPS) and the RNN the CPU, so the two CV runs can go in parallel.
set -euo pipefail
PY="${PY:-python}"

# 1. splits (already in the repository) + audio cache
$PY -m murmur.prepare --config configs/base.yaml

# 2. 5-fold CV on the dev set: one model per fold + out-of-fold predictions
$PY -m murmur.cv --config configs/cnn.yaml
$PY -m murmur.cv --config configs/rnn.yaml

# 3. compare on dev (CV only): the two models and their ensemble
$PY -m murmur.evaluate --exp cnn rnn --dev-only
$PY -m murmur.report

# 4. final evaluation on the held-out test set (once, after all choices are made)
$PY -m murmur.evaluate --exp cnn
$PY -m murmur.evaluate --exp rnn
$PY -m murmur.evaluate --exp cnn rnn

# 5. bonus: retrain on all patients and predict the hidden data (when available)
# $PY -m murmur.cv --config configs/cnn.yaml --all-data
# $PY -m murmur.cv --config configs/rnn.yaml --all-data
# $PY -m murmur.predict --exp cnn_all rnn_all --data-dir data/hidden --out predictions.csv
