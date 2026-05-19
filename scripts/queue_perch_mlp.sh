#!/bin/bash
# queue_perch_mlp.sh — wait for extraction, then train all 5 MLP folds
set -e
cd /data/birdclef2026

LOG="artifacts/logs/perch_mlp_train.log"
EMB_FILE="artifacts/perch/embeddings/clips_embeddings.npy"
META_FILE="artifacts/perch/embeddings/clips_metadata.csv"
SS_EMB_FILE="artifacts/perch/embeddings/ss_embeddings.npy"

echo "[$(date '+%H:%M:%S')] Queue started, waiting for extraction..." | tee "$LOG"

# Wait until extraction is complete (both clips and soundscape embeddings exist)
while true; do
    if [ -f "$EMB_FILE" ] && [ -f "$META_FILE" ] && [ -f "$SS_EMB_FILE" ]; then
        N_CLIPS=$(python3 -c "import numpy as np; a=np.load('$EMB_FILE'); print(len(a))" 2>/dev/null || echo 0)
        if [ "$N_CLIPS" -ge 35000 ]; then
            echo "[$(date '+%H:%M:%S')] Extraction complete: $N_CLIPS clips" | tee -a "$LOG"
            break
        fi
    fi
    echo "[$(date '+%H:%M:%S')] Extraction still running, waiting 60s..." | tee -a "$LOG"
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Starting MLP training: 5 folds, 60 epochs" | tee -a "$LOG"
.venv/bin/python scripts/train_perch_mlp.py \
    --folds 0,1,2,3,4 \
    --epochs 60 \
    --batch-size 512 \
    --lr 3e-3 \
    2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] MLP training complete" | tee -a "$LOG"

# Check if all fold checkpoints exist
N_CKPTS=$(ls artifacts/perch/mlp/fold*_best.pt 2>/dev/null | wc -l)
echo "[$(date '+%H:%M:%S')] Fold checkpoints found: $N_CKPTS" | tee -a "$LOG"

if [ "$N_CKPTS" -ge 5 ]; then
    echo "[$(date '+%H:%M:%S')] Exporting for Kaggle..." | tee -a "$LOG"
    .venv/bin/python scripts/export_perch_for_kaggle.py 2>&1 | tee -a "$LOG"
    
    echo "[$(date '+%H:%M:%S')] Uploading to Kaggle..." | tee -a "$LOG"
    # First-time create vs version
    if kaggle datasets status cid007/birdclef2026-perch 2>/dev/null | grep -q "ready"; then
        kaggle datasets version -p artifacts/exports/perch_mlp -m "perch_mlp_v1" 2>&1 | tee -a "$LOG"
    else
        kaggle datasets create -p artifacts/exports/perch_mlp 2>&1 | tee -a "$LOG"
    fi
    echo "[$(date '+%H:%M:%S')] Kaggle upload done" | tee -a "$LOG"
else
    echo "[$(date '+%H:%M:%S')] WARNING: Only $N_CKPTS folds trained, skipping export" | tee -a "$LOG"
fi

echo "[$(date '+%H:%M:%S')] All done." | tee -a "$LOG"
