#!/usr/bin/env bash
# All experiments of the report, in order. Every run writes outputs/<name>/ and finished runs are skipped,
# so the script can be stopped and restarted.
#
#   bash scripts/experiments.sh                     # 5 folds
#   FOLDS="0 1 2" bash scripts/experiments.sh       # quicker ablations (compare runs with the same folds!)
#   PY=/opt/anaconda3/envs/datasci/bin/python bash scripts/experiments.sh
#
# ~45 min per CNN run (GPU/MPS) and ~60 min per RNN run (CPU) on an Apple M-series laptop.
set -uo pipefail
PY="${PY:-python}"

cv() {  # cv <config> <name> [overrides...]
  local config=$1 name=$2; shift 2
  if [[ -f "outputs/$name/cv_summary.json" ]]; then echo "== $name: done"; return; fi
  echo "== $name"
  local args=(--config "configs/$config.yaml" --name "$name" --resume)
  if [[ -n "${FOLDS:-}" ]]; then args+=(--folds $FOLDS); fi
  if (( $# )); then args+=(--set "$@"); fi
  $PY -m murmur cv "${args[@]}"
}

$PY -m murmur prepare

# 1. baselines
cv cnn cnn
cv rnn rnn

# 2. ablations (dev CV only)
cv cnn cnn_w-none       train.class_weights=none                      # loss without class weights
cv cnn cnn_w-balanced   train.class_weights=balanced                  # inverse-frequency weights
cv cnn cnn_lab-patient  data.label_mode=patient decision.monitor=mean # recordings inherit the patient label
cv cnn cnn_win2s        windows.length_s=2.0 windows.eval_hop_s=1.0
cv cnn cnn_win8s        windows.length_s=8.0 windows.eval_hop_s=4.0
cv cnn cnn_noaug        'train.augment={}' features.spec_augment=null
cv cnn cnn_small        'model.channels=[8,16,32,64]'
cv cnn cnn_big          'model.channels=[32,64,128,256]'
cv rnn rnn_logmel       features.kind=logmel features.n_mels=64
cv rnn rnn_lstm         model.cell=lstm
cv rnn rnn_uni          model.bidirectional=false

# 3. compare on the dev CV: table + figures, and the CNN+RNN ensemble
$PY -m murmur report
$PY -m murmur evaluate --exp cnn rnn --dev-only

# 4. final evaluation on the test set: run ONCE, after every choice is made
#    (replace cnn/rnn by the chosen configurations)
# $PY -m murmur evaluate --exp cnn
# $PY -m murmur evaluate --exp rnn
# $PY -m murmur evaluate --exp cnn rnn

# 5. bonus: retrain on all patients, then predict the hidden data
# $PY -m murmur cv --config configs/cnn.yaml --all-data
# $PY -m murmur cv --config configs/rnn.yaml --all-data
# $PY -m murmur predict --exp cnn_all rnn_all --data-dir ../hidden-data --out predictions.csv
