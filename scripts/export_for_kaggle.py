"""
export_for_kaggle.py — Package trained checkpoints as a Kaggle dataset artifact.

Usage:
    python scripts/export_for_kaggle.py --experiment baseline_v1

Output: artifacts/exports/<experiment>/
    fold0_best.pt ... fold4_best.pt
    train_config.yaml
    label_encoder.json

Upload the output directory as a Kaggle dataset named birdclef2026-model.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_taxonomy_df, build_label_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="baseline_v1")
    p.add_argument("--config", default="configs/train.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp = args.experiment

    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    out_dir = Path(cfg["paths"]["artifacts"]) / "exports" / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy fold checkpoints
    n_copied = 0
    for fold in range(10):
        src = ckpt_dir / f"{exp}_fold{fold}_best.pt"
        if not src.exists():
            break
        dst = out_dir / f"fold{fold}_best.pt"
        shutil.copy2(src, dst)
        print(f"Copied {src.name} → {dst}")
        n_copied += 1

    if n_copied == 0:
        print(f"ERROR: No checkpoints found for experiment '{exp}' in {ckpt_dir}")
        sys.exit(1)

    # Save merged config (base + train)
    config_path = out_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved → {config_path}")

    # Save label encoder (class index → label mapping)
    taxonomy_df = get_taxonomy_df(cfg["paths"]["taxonomy_csv"])
    label_encoder = build_label_encoder(taxonomy_df)
    classes = sorted(label_encoder, key=label_encoder.get)
    le_path = out_dir / "label_encoder.json"
    with open(le_path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"Label encoder saved → {le_path}")

    # Copy src/ so inference.py can import from it on Kaggle
    repo_root = Path(__file__).parent.parent
    src_dst = out_dir / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(repo_root / "src", src_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"src/ copied → {src_dst}")

    # Write dataset-metadata.json for kaggle datasets version
    meta = {"title": "birdclef2026-model", "id": "cid007/birdclef2026-model",
            "licenses": [{"name": "CC0-1.0"}]}
    meta_path = out_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Dataset metadata saved → {meta_path}")

    print(f"\nExport complete: {out_dir}")
    print(f"  {n_copied} fold checkpoints, src/ included")
    print(f"\nTo upload: kaggle datasets version -p {out_dir} -m '{exp}'")
    print("Inference notebook dataset: cid007/birdclef2026-model")


if __name__ == "__main__":
    main()
