#!/usr/bin/env bash
# Steps 2 and 3: feed Uncharted's step 1 output into the original Alien code, unchanged.
# Run from the uncharted repo root after collect, atomize, cluster and build-db.
#
#   bash scripts/train_and_sample.sh            # GPU if available
#   DEVICE=cpu EPOCHS_C=5 EPOCHS_A=3 bash scripts/train_and_sample.sh   # quick CPU smoke run
set -euo pipefail

ROOT="$(pwd)"
DATA="$ROOT/data"
MODELS="$ROOT/models"
ALIEN="$ROOT/third_party/alien-space-of-science"
DEVICE="${DEVICE:-cuda}"
EPOCHS_C="${EPOCHS_C:-30}"
EPOCHS_A="${EPOCHS_A:-100}"
BATCH_A="${BATCH_A:-4096}"
BETA="${BETA:-0.7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ ! -d "$ALIEN" ]; then
  git clone https://github.com/alejandrohdez00/alien-space-of-science.git "$ALIEN"
fi
cd "$ALIEN"

python alien.py make-datasets --clusters "$DATA/clusters_80.json" --vocab "$DATA/vocab.json" \
  --split 0.9 --output-dir "$DATA/coherence_dataset"

python alien.py make-availability --clusters "$DATA/clusters_80.json" --db "$DATA/uncharted.db" \
  --token-mapping "$DATA/coherence_dataset/token_mapping.json" \
  --output-dir "$DATA/availability_dataset"

python alien.py train-coherence \
  --train "$DATA/coherence_dataset/coherence_train.jsonl" \
  --val "$DATA/coherence_dataset/coherence_val.jsonl" \
  --token-mapping "$DATA/coherence_dataset/token_mapping.json" \
  --output-dir "$MODELS/coherence_model" \
  --epochs "$EPOCHS_C" --batch-size 64 --gradient-accumulation-steps 2 \
  --learning-rate 0.0001 --warmup-steps 1000 --max-seq-len 32

python alien.py train-availability --dataset "$DATA/availability_dataset" \
  --output-dir "$MODELS/availability_model" --batch-size "$BATCH_A" \
  --epochs "$EPOCHS_A" --device "$DEVICE"

python alien.py sample \
  --coherence-model "$MODELS/coherence_model" \
  --availability-model "$MODELS/availability_model" \
  --betas "$BETA" --top-k 50 --device "$DEVICE" --output-dir "$ROOT/results/alien"

echo "Done. Top alien combinations: $ROOT/results/alien"
echo "Readable version: python $ROOT/uncharted.py show --results $ROOT/results/alien/k3/beta_$BETA/results.json"
