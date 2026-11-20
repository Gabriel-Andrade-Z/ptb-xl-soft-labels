"""Train (or run inference with) one experiment of the paper from a JSON config.

Usage:
    python train.py --config configs/bce_soft.json --data /path/to/dataset_split
    python train.py --config configs/bce_hard.json --data /path/to/dataset_split --seeds 42
    python train.py --config configs/ablation/B_se_fft.json --data /path/to/dataset_split
    python train.py infer --config configs/bce_soft.json --data /path/to/dataset_split --seed 42

<data> must contain train/, val/ and test/ with one <ecg_id>.pt file per ECG
(see README). Outputs go to <out>/<config name>/<arch>_<loss>/seed_<s>/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Config  # noqa: E402
import pipeline  # noqa: E402


def carregar_config(caminho: Path, data: str, out: str, seeds: list[int] | None) -> Config:
    """Build a Config from the JSON file; keys starting with '_' are comments."""
    bruto = json.loads(caminho.read_text())
    campos = {f.name for f in dataclasses.fields(Config)}
    desconhecidos = [k for k in bruto if not k.startswith("_") and k not in campos]
    if desconhecidos:
        raise ValueError(f"unknown keys in {caminho}: {desconhecidos}")
    kw = {k: v for k, v in bruto.items() if not k.startswith("_")}
    kw.update(caminho=data, out_dir=str(Path(out) / caminho.stem))
    if seeds:
        kw["seeds"] = seeds
    return Config(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["train", "infer"], default="train")
    ap.add_argument("--config", required=True, type=Path, help="JSON file in configs/")
    ap.add_argument("--data", required=True, help="folder with train/, val/, test/")
    ap.add_argument("--out", default="runs", help="output root (default: runs/)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds of the config")
    ap.add_argument("--seed", type=int, help="infer: seed of the trained model")
    ap.add_argument("--n", type=int, default=5, help="infer: number of examples")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="infer: partition")
    a = ap.parse_args()

    cfg = carregar_config(a.config, a.data, a.out, a.seeds)
    print(f"config: {a.config} | arch={cfg.arch_type} | loss={cfg.loss_type} | dropout={cfg.dropout_cnn} | seeds={cfg.seeds}")
    if a.cmd == "train":
        pipeline.treinar(cfg)
    else:
        pipeline.inferir(cfg, a.seed if a.seed is not None else cfg.seeds[0], a.n, a.split)


if __name__ == "__main__":
    main()
