"""Configuracao central do pipeline PTB-XL (5 superclasses, Modelo A ou Modelo B).

O particionamento por paciente ja foi feito fisicamente em disco:
    <caminho>/train/*.pt, <caminho>/val/*.pt, <caminho>/test/*.pt
Os experimentos da revisao sao selecionados aqui: arquitetura (Modelo A/B), tipo
de perda, lista de seeds e temperature scaling. Os hiperparametros de treino sao
os mesmos do artigo."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

CLASSES_ORIG = ["NORM", "MI", "STTC", "CD", "HYP"]
ORDEM = [1, 2, 3, 4, 0]  # prioridade de risco: MI, STTC, CD, HYP, NORM
CLASSES = [CLASSES_ORIG[i] for i in ORDEM]

LOSS_TYPES = ("bce_hard", "bce_soft", "label_smoothing", "focal_loss")
ARCH_TYPES = ("shared", "specialists")  # Modelo B | Modelo A
PARTICOES = ("train", "val", "test")  # nomes das subpastas dentro de `caminho`


@dataclass
class Config:
    # ------------------------------------------------------------------ dados
    caminho: str = "data/dataset_split"  # overridden by train.py --data
    out_dir: str = "resultados"
    n_leads: int = 12

    # ------------------------------------------------------------ multi-seed
    # O split e fixo (pastas); a seed controla inicializacao, ordem dos lotes,
    # augmentation e dropout. A variancia entre seeds e, portanto, de treino.
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    deterministic: bool = False      # o artigo usou cudnn.benchmark=True (nao deterministico)

    # ------------------------------------------------------------- arquitetura
    # "shared"      : Modelo B, backbone unico + 5 saidas (lr = lr_cnn)
    # "specialists" : Modelo A, 5 SpecialistResNet independentes (lr = lr_cnn_por_classe)
    arch_type: str = "shared"

    # ------------------------------------------------------------------ treino
    batch_size: int = 256
    num_workers: int = 4
    lr_cnn: float = 5e-4
    lr_cnn_por_classe: dict[str, float] = field(default_factory=lambda: {
        "MI": 4e-4, "STTC": 3e-4, "CD": 2e-4, "HYP": 5e-4, "NORM": 2e-4,
    })
    epochs_cnn: int = 60
    weight_decay: float = 2e-3
    dropout_cnn: float = 0.15
    es_patience: int = 12
    warmup: int = 12
    grad_clip: float = 3.0
    es_suavizacao: float = 0.85      # media movel exponencial do val_loss para o early stop

    # peso do positivo por classe; identico em todos os regimes de perda (HYP e rara)
    pos_weight: dict[str, float] = field(default_factory=lambda: {
        "MI": 1.0, "STTC": 1.0, "CD": 1.0, "HYP": 3.0, "NORM": 1.0,
    })

    # presenca = confiabilidade normalizada >= 0.5 (>= 50% na escala original 0-100)
    limiar_presenca: float = 0.5

    # ----------------------------------------------------- ablacao arquitetural
    usar_se: bool = True
    usar_fft: bool = True
    usar_attn: bool = False

    # ------------------------------------------------------------------- perda
    # bce_hard        : BCE com alvo binarizado em limiar_presenca
    # bce_soft        : BCE com a confiabilidade continua do cardiologista (proposta)
    # label_smoothing : alvo binario suavizado para {eps/2, 1-eps/2}
    # focal_loss      : focal multirrotulo sobre o alvo binario
    loss_type: str = "bce_soft"
    ls_epsilon: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: float | None = None  # None = sem balanceamento alpha (pos_weight ja trata HYP)

    # ------------------------------------------------------------- calibracao
    # ajusta um T escalar no val (L-BFGS) e avalia o teste com e sem escala
    apply_temperature_scaling: bool = False
    ts_init: float = 1.5

    summary_depth: int = 2
    summary_max_lines: int = 30

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    def __post_init__(self) -> None:
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type deve ser um de {LOSS_TYPES}, recebido {self.loss_type!r}")
        if self.arch_type not in ARCH_TYPES:
            raise ValueError(f"arch_type deve ser um de {ARCH_TYPES}, recebido {self.arch_type!r}")
        if not self.seeds:
            raise ValueError("seeds nao pode ser vazia")