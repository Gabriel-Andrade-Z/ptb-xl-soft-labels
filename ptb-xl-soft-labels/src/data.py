"""Dataset de ECG e leitura das particoes pre-separadas em disco.

O split por paciente foi feito previamente e gravado em subpastas
(<caminho>/train, <caminho>/val, <caminho>/test); aqui apenas listamos os .pt
de cada uma. Os caminhos sao completos, entao o ECGDataset nao depende de uma
pasta base unica."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CLASSES, ORDEM, PARTICOES, Config


class ECGDataset(Dataset):
    """ECG de 12 derivacoes (5000 amostras, 500 Hz) e a confiabilidade diagnostica
    continua das 5 superclasses, normalizada para [0,1] e reordenada por risco."""
    def __init__(self, arquivos: list[str], augment: bool = False):
        self.arquivos, self.augment = arquivos, augment  # caminhos completos dos .pt

    def __len__(self) -> int:
        return len(self.arquivos)

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.15:                       # ruido gaussiano
            x = x + 0.01 * torch.randn_like(x)
        if torch.rand(1).item() < 0.15:                       # baseline wander
            t = torch.linspace(0, 1, x.size(-1))
            f = 0.15 + 0.15 * torch.rand(1).item()
            x = x + 0.05 * torch.sin(2 * np.pi * f * t * (x.size(-1) / 500.0))
        if torch.rand(1).item() < 0.10:                       # lead dropout suave
            lead = int(torch.randint(0, x.size(0), (1,)).item())
            x[lead] = x[lead] * 0.3
        return x

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, c = torch.load(self.arquivos[i], map_location="cpu")
        x = x.float()
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        c = (c.float() / 100.0)[ORDEM]
        if self.augment:
            x = self._aug(x)
        return x, c


def listar_particoes(cfg: Config) -> dict[str, list[str]]:
    """Retorna {"train": [...], "val": [...], "test": [...]} com os caminhos completos
    dos .pt de cada subpasta, em ordem lexicografica (ordem estavel entre execucoes).

    Falha se alguma pasta estiver ausente/vazia ou se o mesmo nome de arquivo (mesmo
    exame) aparecer em mais de uma particao, o que indicaria erro na separacao."""
    particoes = {}
    for nome in PARTICOES:
        pasta = os.path.join(cfg.caminho, nome)
        if not os.path.isdir(pasta):
            raise FileNotFoundError(f"Pasta da particao '{nome}' nao encontrada: {pasta}")
        arqs = sorted(f for f in os.listdir(pasta) if f.endswith(".pt"))
        if not arqs:
            raise FileNotFoundError(f"Nenhum .pt em {pasta}")
        particoes[nome] = [os.path.join(pasta, f) for f in arqs]

    nomes = [set(map(os.path.basename, v)) for v in particoes.values()]
    repetidos = (nomes[0] & nomes[1]) | (nomes[0] & nomes[2]) | (nomes[1] & nomes[2])
    if repetidos:
        raise RuntimeError(f"{len(repetidos)} arquivos em mais de uma particao (ex.: {sorted(repetidos)[:5]})")

    print(f"Particoes lidas de {cfg.caminho}: " +
          " | ".join(f"{k}: {len(v)}" for k, v in particoes.items()))
    return particoes


def prevalencia(arquivos: list[str], limiar: float) -> np.ndarray:
    """Fracao de exames com presenca (conf >= limiar) por classe, na ordem CLASSES.
    Le apenas os rotulos; custo de uma passada pelos arquivos."""
    conf = np.stack([(torch.load(f, map_location="cpu")[1].float().numpy() / 100.0)[ORDEM]
                     for f in arquivos])
    return (conf >= limiar).mean(0)


def resumir_particoes(particoes: dict[str, list[str]], cfg: Config) -> dict:
    """Imprime e retorna tamanho e prevalencia por classe de cada particao
    (tabela de caracterizacao do conjunto para o artigo)."""
    print(f"{'part.':<7}{'regs':>7}  " + "".join(f"{c:>7}" for c in CLASSES))
    resumo = {}
    for nome, arqs in particoes.items():
        prev = prevalencia(arqs, cfg.limiar_presenca)
        resumo[nome] = {"registros": len(arqs), "prevalencia": dict(zip(CLASSES, np.round(prev, 4).tolist()))}
        print(f"{nome:<7}{len(arqs):>7}  " + "".join(f"{v:>7.3f}" for v in prev))
    return resumo