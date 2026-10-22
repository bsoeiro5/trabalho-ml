#!/usr/bin/env bash
# Ablation studies on the dev CV (never on test). Each run writes outputs/<name>/.
#
#   bash scripts/run_experiments.sh            # all experiments, 5 folds
#   FOLDS="0 1 2" bash scripts/run_experiments.sh   # quicker, 3 folds (compare runs with the same folds!)
#
# Results: python -m murmur.report  (table in outputs/report/cv_table.md)
set -uo pipefail
PY="${PY:-python}"
run() {  # run <config> <name> [overrides...]
  local config=$1 name=$2; shift 2
  if [[ -f "outputs/$name/cv_summary.json" ]]; then echo "== $name: done"; return; fi
  echo "== $name"
  local args=(--config "configs/$config.yaml" --name "$name" --resume)
  if [[ -n "${FOLDS:-}" ]]; then args+=(--folds $FOLDS); fi
  if (( $# )); then args+=(--set "$@"); fi
  $PY -m murmur.cv "${args[@]}"
}

# baselines
run cnn cnn
run rnn rnn

# loss weights (class imbalance): metric weights (5,3,1) vs none vs inverse frequency
run cnn cnn_w-none       train.class_weights=none
run cnn cnn_w-balanced   train.class_weights=balanced

# recording labels: from "Murmur locations" vs copy the patient label
run cnn cnn_lab-patient  data.label_mode=patient decision.monitor=mean

# window length
run cnn cnn_win2s        windows.length_s=2.0 windows.eval_hop_s=1.0
run cnn cnn_win8s        windows.length_s=8.0 windows.eval_hop_s=4.0

# regularisation: no data augmentation
run cnn cnn_noaug        'train.augment={}' features.spec_augment=null

# capacity
run cnn cnn_small        'model.channels=[8,16,32,64]'
run cnn cnn_big          'model.channels=[32,64,128,256]'

# RNN: features, cell type, direction
run rnn rnn_logmel       features.kind=logmel features.n_mels=64
run rnn rnn_lstm         model.cell=lstm
run rnn rnn_uni          model.bidirectional=false

$PY -m murmur.report
