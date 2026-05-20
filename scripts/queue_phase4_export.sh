#!/bin/bash
# queue_phase4_export.sh — wait for fold4, export all 5 folds, upload, push ensemble kernel
set -e
cd /data/birdclef2026

LOG="artifacts/logs/phase4_export.log"
FOLD4_CKPT="artifacts/checkpoints/phase4_ls_fold4_best.pt"
FOLD4_DONE_MARKER="artifacts/checkpoints/.phase4_fold4_done"

echo "[$(date '+%H:%M:%S')] Waiting for phase4 fold4..." | tee "$LOG"

# Wait for fold4 training to complete — detect via log "Fold 4 complete"
until grep -q "Fold 4 complete" artifacts/logs/phase4_ls.log 2>/dev/null; do
    echo "[$(date '+%H:%M:%S')] fold4 still training..." | tee -a "$LOG"
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Fold 4 complete! Exporting..." | tee -a "$LOG"

# Log fold 4 best AUC
grep "Fold 4 best macro AUC\|fold4_best" artifacts/logs/phase4_ls.log | tail -3 | tee -a "$LOG"

# Export all 5 folds
.venv/bin/python scripts/export_for_kaggle.py \
    --experiment phase4_ls \
    --config configs/train_phase4_ls.yaml \
    2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] Export done. Uploading to Kaggle..." | tee -a "$LOG"

# Upload dataset (version existing or create new)
if kaggle datasets status cid007/birdclef2026-model 2>/dev/null | grep -q "ready"; then
    kaggle datasets version -p artifacts/exports/phase4_ls -m "phase4_ls_5fold" \
        2>&1 | tee -a "$LOG"
else
    kaggle datasets create -p artifacts/exports/phase4_ls \
        2>&1 | tee -a "$LOG"
fi

echo "[$(date '+%H:%M:%S')] Upload done. Pushing ensemble kernel..." | tee -a "$LOG"

# Push ensemble kernel
cp kaggle_notebook/inference_ensemble.py kaggle_kernel_ensemble/inference_ensemble.py
kaggle kernels push -p kaggle_kernel_ensemble 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] Ensemble kernel pushed." | tee -a "$LOG"

# Also push the phase4-only kernel for LB comparison
cp kaggle_notebook/inference.py kaggle_notebook/inference_phase4.py 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] All done." | tee -a "$LOG"
