"""Materialize the patient-level split used in the paper.

Usage:
    python scripts/materialize_split.py <tensors_dir> <out_dir> [--csv split_patient_level.csv] [--link]

<tensors_dir> holds one <ecg_id:05d>.pt file per PTB-XL record (e.g. 00001.pt), each a tuple
(x, c): x = float tensor (12, 5000) with the 500 Hz signal, c = float tensor (5,) with the
cardiologist reliability (0-100) in the order NORM, MI, STTC, CD, HYP.
The files are copied (or symlinked with --link) into <out_dir>/{train,val,test}/.
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("tensors_dir", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[1] / "split_patient_level.csv")
ap.add_argument("--link", action="store_true", help="create symlinks instead of copying")
a = ap.parse_args()

split = pd.read_csv(a.csv)
faltando = [i for i in split.ecg_id if not (a.tensors_dir / f"{i:05d}.pt").exists()]
if faltando:
    raise FileNotFoundError(f"{len(faltando)} tensors missing in {a.tensors_dir} (e.g. {faltando[:5]})")

for part, grupo in split.groupby("particao"):
    destino = a.out_dir / part
    destino.mkdir(parents=True, exist_ok=True)
    for i in grupo.ecg_id:
        src, dst = a.tensors_dir / f"{i:05d}.pt", destino / f"{i:05d}.pt"
        if dst.exists():
            continue
        os.symlink(src.resolve(), dst) if a.link else shutil.copy2(src, dst)
    print(f"{part:<6} {len(grupo):>6} ECGs  {grupo.patient_id.nunique():>6} patients -> {destino}")

pac = {p: set(g.patient_id) for p, g in split.groupby("particao")}
assert not (pac["train"] & pac["val"] or pac["train"] & pac["test"] or pac["val"] & pac["test"]), "patient leakage"
print("no patient shared between subsets")
