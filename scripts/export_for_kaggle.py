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
    # label_encoder is {primary_label: class_index} — sort by index for ordered list
    classes = sorted(label_encoder, key=label_encoder.get)
    le_path = out_dir / "label_encoder.json"
    with open(le_path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"Label encoder saved → {le_path}")

    print(f"\nExport complete: {out_dir}")
    print(f"  {n_copied} fold checkpoints")
    print(f"\nUpload '{out_dir}' as a Kaggle dataset named: birdclef2026-model")
    print("Then set paths.model_dir = /kaggle/input/birdclef2026-model in inference notebook.")


if __name__ == "__main__":
    main()
