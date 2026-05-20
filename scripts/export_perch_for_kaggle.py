"""
export_perch_for_kaggle.py — Package Perch MLP checkpoints as self-contained Kaggle dataset.

The output dataset (cid007/birdclef2026-perch) contains:
  perch_v2.onnx                 — Perch feature extractor (self-owned copy, no 3rd-party dependency)
  fold{k}_mlp_best.pt           — MLP classifier weights for each fold
  label_encoder.json            — class list (234 classes)
  src/perch_mlp.py              — MLP architecture (needed by inference.py)
  dataset-metadata.json

Usage:
  cd /data/birdclef2026
  .venv/bin/python scripts/export_perch_for_kaggle.py
  kaggle datasets version -p artifacts/exports/perch_mlp -m "perch_mlp_v1"
"""

import json
import shutil
import sys
from pathlib import Path

REPO_DIR  = Path(__file__).parent.parent
MLP_DIR   = REPO_DIR / "artifacts" / "perch" / "mlp"
ONNX_SRC  = REPO_DIR / "artifacts" / "perch" / "perch_v2.onnx"
LE_SRC    = REPO_DIR / "artifacts" / "exports" / "phase4_ls" / "label_encoder.json"
OUT_DIR   = REPO_DIR / "artifacts" / "exports" / "perch_mlp"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Perch ONNX
    onnx_dst = OUT_DIR / "perch_v2.onnx"
    if not onnx_dst.exists():
        shutil.copy2(ONNX_SRC, onnx_dst)
        print(f"Copied {ONNX_SRC.name} → {onnx_dst}  ({ONNX_SRC.stat().st_size//1_000_000}MB)")
    else:
        print(f"perch_v2.onnx already in export dir ({onnx_dst.stat().st_size//1_000_000}MB)")

    # 2. MLP checkpoints
    n_folds = 0
    for fold in range(10):
        src = MLP_DIR / f"fold{fold}_best.pt"
        if not src.exists():
            break
        dst = OUT_DIR / f"fold{fold}_mlp_best.pt"
        shutil.copy2(src, dst)
        print(f"Copied {src.name} → {dst.name}")
        n_folds += 1

    if n_folds == 0:
        print("WARNING: No MLP checkpoints found. Train first with scripts/train_perch_mlp.py")

    # 3. Label encoder
    shutil.copy2(LE_SRC, OUT_DIR / "label_encoder.json")
    print(f"Copied label_encoder.json")

    # 4. onnxruntime wheel (Kaggle CPU doesn't have it pre-installed)
    whl_src = REPO_DIR / "artifacts" / "perch" / "onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    if whl_src.exists():
        shutil.copy2(whl_src, OUT_DIR / whl_src.name)
        print(f"Copied {whl_src.name} ({whl_src.stat().st_size//1_000_000}MB)")
    else:
        print(f"WARNING: onnxruntime wheel not found at {whl_src}")

    # 5. src/perch_mlp.py — zip it so kaggle datasets version includes it
    import zipfile
    zip_path = OUT_DIR / "src.zip"
    mlp_src = REPO_DIR / "src" / "perch_mlp.py"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(mlp_src), arcname="src/perch_mlp.py")
    print(f"Created src.zip ({zip_path.stat().st_size} bytes, contains: {[i.filename for i in zipfile.ZipFile(str(zip_path)).infolist()]})")

    # 5. Dataset metadata
    meta = {
        "title": "birdclef2026-perch",
        "id": "cid007/birdclef2026-perch",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(OUT_DIR / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExport complete: {OUT_DIR}")
    print(f"  Folds: {n_folds}")
    print(f"\nTo upload:")
    print(f"  kaggle datasets version -p {OUT_DIR} -m 'perch_mlp_v1'")
    print(f"  (first time: kaggle datasets create -p {OUT_DIR})")


if __name__ == "__main__":
    main()
