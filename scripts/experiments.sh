#!/usr/bin/env bash
# All experiments of the report, in order. Every run writes outputs/<name>/ and finished runs are
# skipped, so the script can be stopped and restarted.
#
#   bash scripts/experiments.sh cnn     # CNN experiments (GPU)  } run both at the same time,
#   bash scripts/experiments.sh rnn     # RNN experiments (CPU)  } in two terminals
#   bash scripts/experiments.sh         # everything, one after the other
#   FOLDS="0 1 2" bash scripts/experiments.sh cnn         # quicker (compare runs with the same folds!)
#   PY=/opt/anaconda3/envs/datasci/bin/python bash scripts/experiments.sh cnn
#
# Approximate times on an Apple M-series laptop: ~30 min per CNN run, ~60 min per RNN run.
set -uo pipefail
PY="${PY:-python}"
GROUP="${1:-all}"

cv() {  # cv <config> <name> [overrides...]
  local config=$1 name=$2; shift 2
  if [[ -f "outputs/$name/cv_summary.json" ]]; then echo "== $name: done"; return; fi
  echo "== $name"
  local args=(--config "configs/$config.yaml" --name "$name" --resume)
  if [[ -n "${FOLDS:-}" ]]; then args+=(--folds $FOLDS); fi
  if (( $# )); then args+=(--set "$@"); fi
  $PY -m murmur cv "${args[@]}"
}

$PY -m murmur prepare > /dev/null

if [[ $GROUP != rnn ]]; then
  # phase 1: training recipe
  cv cnn cnn_es          train.epochs=60 train.early_stopping=wa train.ema=0     # early stopping on the validation WA
  cv cnn cnn_es-ema      train.epochs=60 train.early_stopping=wa                 # + EMA of the weights
  cv cnn cnn_fixed30     train.epochs=30                                         # fixed schedule (30 epochs) + EMA
  cv cnn cnn                                                                     # fixed schedule (20 epochs) + EMA = base
  # phase 2: ablations of the base CNN
  cv cnn cnn_w-none       train.class_weights=none                       # loss without class weights
  cv cnn cnn_w-balanced   train.class_weights=balanced                   # inverse-frequency weights
  cv cnn cnn_lab-patient  data.label_mode=patient                        # recordings inherit the patient label
  cv cnn cnn_win2s        windows.length_s=2.0 windows.eval_hop_s=1.0
  cv cnn cnn_win8s        windows.length_s=8.0 windows.eval_hop_s=4.0
  cv cnn cnn_noaug        'train.augment={}' features.spec_augment=null
  cv cnn cnn_small        'model.channels=[8,16,32,64]'
  cv cnn cnn_big          'model.channels=[32,64,128,256]'
fi

if [[ $GROUP != cnn ]]; then
  cv rnn rnn_es          train.epochs=60 train.early_stopping=wa train.ema=0
  cv rnn rnn_fixed30     train.epochs=30
  cv rnn rnn                                                                     # base
  cv rnn rnn_logmel       features.kind=logmel features.n_mels=64
  cv rnn rnn_lstm         model.cell=lstm
  cv rnn rnn_uni          model.bidirectional=false
fi

if [[ $GROUP == all ]]; then
  $PY -m murmur report                               # table + figures (dev CV only)
  $PY -m murmur evaluate --exp cnn rnn --dev-only    # CNN+RNN ensemble (dev CV only)
fi

# Final evaluation on the test set: run ONCE, after every choice is made (replace by the chosen runs):
#   $PY -m murmur evaluate --exp cnn
#   $PY -m murmur evaluate --exp rnn
# Bonus: retrain on all patients, then predict the hidden data:
#   $PY -m murmur cv --config configs/cnn.yaml --all-data
#   $PY -m murmur predict --exp cnn_all --data-dir ../hidden-data --out predictions.csv
