"""Conta pacientes unicos por particao e confirma ausencia de vazamento.

Uso: python contar_pacientes_por_particao.py <ptbxl_database.csv> <dataset_split>
Gera split_patient_level.csv (ecg_id, patient_id, particao) para publicar junto
com o codigo (resposta ao revisor 8) e imprime os numeros da Tabela de split."""
import os
import sys

import pandas as pd

meta = pd.read_csv(sys.argv[1], usecols=["ecg_id", "patient_id"])
linhas = []
for part in ("train", "val", "test"):
    for f in os.listdir(os.path.join(sys.argv[2], part)):
        if f.endswith(".pt"):
            linhas.append({"ecg_id": int(os.path.splitext(f)[0]), "particao": part})
split = pd.DataFrame(linhas).merge(meta, on="ecg_id", how="left", validate="one_to_one")
assert split.patient_id.notna().all(), "ecg_id sem patient_id no CSV"
split["patient_id"] = split.patient_id.astype(int)

pac = {p: set(g.patient_id) for p, g in split.groupby("particao")}
print(f"{'particao':<8}{'ECGs':>8}{'pacientes':>11}")
for p in ("train", "val", "test"):
    print(f"{p:<8}{(split.particao == p).sum():>8}{len(pac[p]):>11}")
print(f"{'total':<8}{len(split):>8}{split.patient_id.nunique():>11}")
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    print(f"pacientes em comum {a}/{b}: {len(pac[a] & pac[b])}")
split.sort_values("ecg_id").to_csv("split_patient_level.csv", index=False)
print("-> split_patient_level.csv")
