"""Pipeline multi-seed sobre particoes fixas em disco: treino -> (temperature scaling) -> teste.

Uso:
    python main.py                                   # config.py como esta
    python main.py --loss bce_hard                   # baseline hard label
    python main.py --loss bce_soft --ts              # + temperature scaling
    python main.py --loss focal_loss --seeds 42 43   # subconjunto de seeds
    python main.py treinar --arch specialists --loss bce_hard   # Modelo A
    python main.py inferir --loss bce_soft --seed 42 --n 5 --particao test

Saidas em <out_dir>/<arch_type>_<loss_type>/:
    config.json, particoes.json (tamanho e prevalencia de train/val/test),
    resumo.json (media/desvio/IC95 entre seeds), resumo_por_seed.csv,
    seed_<s>/{pesos, limiares*.npy, temperatura.npy, metricas.json, logits.npz, graficos}
    pesos = modelo.pt (shared) ou esp_<CLASSE>.pt x5 (specialists).
Os dois modelos produzem logits (N,5) na mesma ordem CLASSES, entao avaliacao,
limiares e temperature scaling sao exatamente os mesmos. Como o split e fixo,
execucoes com a mesma seed podem ser comparadas de forma pareada exame a exame."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from config import ARCH_TYPES, CLASSES, LOSS_TYPES, Config
from data import ECGDataset, listar_particoes, resumir_particoes
from engine import (TemperatureScaler, avaliar_calibracao, avaliar_multilabel, avaliar_regressao,
                    otimizar_limiares, plot_confusao_multilabel, treinar_compartilhado,
                    treinar_especialista)
from models import SharedBackboneNet, SharedLoss, SpecialistLoss, SpecialistResNet, print_summary

# Modelo B = uma rede (N,5); Modelo A = lista de 5 especialistas (N,1), na ordem CLASSES
Modelo = SharedBackboneNet | list[SpecialistResNet]

# metricas macro agregadas entre seeds (F1* = F1 nos limiares otimizados no val, como no artigo)
METRICAS = {"auc_macro": "AUC", "ap_macro": "AUPRC", "f1_macro_opt": "F1*", "f1_macro_05": "F1@0.5",
            "ece_macro": "ECE", "brier_macro": "Brier", "mae_macro": "MAE", "rmse_macro": "RMSE"}


def dir_experimento(cfg: Config) -> str:
    return os.path.join(cfg.out_dir, f"{cfg.arch_type}_{cfg.loss_type}")


def fixar_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # tambem semeia todas as GPUs
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def construir_modelo(cfg: Config) -> Modelo:
    """SharedBackboneNet (Modelo B) ou lista de 5 SpecialistResNet (Modelo A)."""
    kw = dict(n_leads=cfg.n_leads, drop=cfg.dropout_cnn,
              usar_se=cfg.usar_se, usar_fft=cfg.usar_fft, usar_attn=cfg.usar_attn)
    if cfg.arch_type == "specialists":
        return [SpecialistResNet(**kw) for _ in CLASSES]
    return SharedBackboneNet(n_classes=len(CLASSES), **kw)


def construir_perda(cfg: Config, idx: int | None = None) -> SharedLoss:
    """SharedLoss das 5 saidas (idx=None) ou SpecialistLoss da classe CLASSES[idx];
    mesmos regimes e hiperparametros nos dois casos."""
    kw = dict(loss_type=cfg.loss_type, limiar=cfg.limiar_presenca, ls_epsilon=cfg.ls_epsilon,
              focal_gamma=cfg.focal_gamma, focal_alpha=cfg.focal_alpha)
    if idx is not None:
        return SpecialistLoss(idx, cfg.pos_weight.get(CLASSES[idx], 1.0), **kw)
    return SharedLoss([cfg.pos_weight.get(c, 1.0) for c in CLASSES], **kw)


def arquivos_pesos(cfg: Config, pasta: str) -> list[str]:
    """modelo.pt (shared) ou esp_<CLASSE>.pt na ordem CLASSES (specialists)."""
    if cfg.arch_type == "specialists":
        return [os.path.join(pasta, f"esp_{nome}.pt") for nome in CLASSES]
    return [os.path.join(pasta, "modelo.pt")]


def _redes(modelo: Modelo) -> list[torch.nn.Module]:
    return modelo if isinstance(modelo, list) else [modelo]


def forward_logits(modelo: Modelo, x: torch.Tensor) -> torch.Tensor:
    """Logits (N,5) em ambos os casos: especialistas concatenados na ordem CLASSES."""
    if isinstance(modelo, list):
        return torch.cat([m(x) for m in modelo], dim=1)
    return modelo(x)


def criar_loaders(cfg: Config, particoes: dict[str, list[str]]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders de treino (com augmentation, embaralhado), validacao e teste."""
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=(cfg.device.type == "cuda"))
    return (DataLoader(ECGDataset(particoes["train"], augment=True), shuffle=True, drop_last=True, **kw),
            DataLoader(ECGDataset(particoes["val"]), shuffle=False, **kw),
            DataLoader(ECGDataset(particoes["test"]), shuffle=False, **kw))


@torch.no_grad()
def extrair_logits(modelo: Modelo, ld: DataLoader, cfg: Config):
    """Retorna logits (N,5), alvos_bin (N,5) = presenca (c >= limiar) e
    alvos_cont (N,5) = confiabilidade do cardiologista em [0,1]."""
    for m in _redes(modelo):
        m.eval()
    L, B, C = [], [], []
    for x, c in ld:
        L.append(forward_logits(modelo, x.to(cfg.device)).cpu())
        B.append((c >= cfg.limiar_presenca).float())
        C.append(c)
    return torch.cat(L).numpy(), torch.cat(B).numpy(), torch.cat(C).numpy()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.as_tensor(z)).numpy()


def avaliar(P: np.ndarray, B: np.ndarray, C: np.ndarray, limiares: np.ndarray, tag: str, pasta: str) -> dict:
    """Roda as quatro avaliacoes originais e junta as metricas macro num dict."""
    t = tag.lower()
    res = avaliar_multilabel(P, B, f"{tag} — Multi-rotulo", os.path.join(pasta, f"ml_{t}"), limiares)
    plot_confusao_multilabel(P, B, limiares, f"Matriz de Confusao ({tag})", os.path.join(pasta, f"mc_{t}.png"))
    res |= avaliar_regressao(P, C, tag)
    res |= avaliar_calibracao(P, B, tag, os.path.join(pasta, f"calib_{t}.png"))
    print(f"\n>>> [{tag}] " + " | ".join(f"{METRICAS[k]} {res[k]:.4f}" for k in METRICAS))
    return res


def rodar_seed(cfg: Config, particoes: dict[str, list[str]], seed: int, mostrar_sumario: bool) -> dict:
    """Uma realizacao completa do experimento para uma seed (split fixo)."""
    t0 = time.time()
    pasta = os.path.join(dir_experimento(cfg), f"seed_{seed}")
    os.makedirs(pasta, exist_ok=True)
    print(f"\n{'#'*64}\n SEED {seed} | arch={cfg.arch_type} | loss={cfg.loss_type}\n{'#'*64}")

    fixar_seed(seed, cfg.deterministic)
    ld_tr, ld_vl, ld_te = criar_loaders(cfg, particoes)

    modelo = construir_modelo(cfg)
    if mostrar_sumario:  # no Modelo A, os 5 especialistas sao identicos: mostra o primeiro
        print_summary(_redes(modelo)[0], ld_tr, cfg.device, cfg.summary_depth, cfg.summary_max_lines)
        n_par = sum(p.numel() for m in _redes(modelo) for p in m.parameters() if p.requires_grad)
        print(f"Parametros treinaveis ({cfg.arch_type}): {n_par:,}")

    if cfg.arch_type == "shared":
        treinar_compartilhado(modelo, ld_tr, ld_vl, construir_perda(cfg).to(cfg.device), cfg,
                              tag=f"seed {seed}")
    else:
        for idx, nome in enumerate(CLASSES):
            treinar_especialista(modelo[idx], ld_tr, ld_vl, construir_perda(cfg, idx).to(cfg.device), cfg,
                                 lr=cfg.lr_cnn_por_classe[nome], tag=f"seed {seed} | especialista {nome}")
    for m, arq in zip(_redes(modelo), arquivos_pesos(cfg, pasta)):
        torch.save(m.state_dict(), arq)

    Lvl, Bvl, Cvl = extrair_logits(modelo, ld_vl, cfg)
    Lte, Bte, Cte = extrair_logits(modelo, ld_te, cfg)
    Pvl, Pte = _sigmoid(Lvl), _sigmoid(Lte)

    limiares = otimizar_limiares(Pvl, Bvl)
    np.save(os.path.join(pasta, "limiares.npy"), limiares)
    print("\nLimiares (val):", " | ".join(f"{c}:{t:.2f}" for c, t in zip(CLASSES, limiares)))

    res = {"seed": seed, "limiares": limiares.tolist(),
           "val": avaliar(Pvl, Bvl, Cvl, limiares, "VAL", pasta),
           "teste": avaliar(Pte, Bte, Cte, limiares, "TESTE", pasta)}
    extras = {}

    if cfg.apply_temperature_scaling:
        ts = TemperatureScaler(cfg.ts_init)
        res["T"] = ts.ajustar(Lvl, Bvl)
        Pvl_ts, Pte_ts = _sigmoid(ts.escalar(Lvl)), _sigmoid(ts.escalar(Lte))
        limiares_ts = otimizar_limiares(Pvl_ts, Bvl)  # limiares reotimizados nas probs escaladas
        np.save(os.path.join(pasta, "temperatura.npy"), np.array([res["T"]]))
        np.save(os.path.join(pasta, "limiares_ts.npy"), limiares_ts)
        res["limiares_ts"] = limiares_ts.tolist()
        res["teste_ts"] = avaliar(Pte_ts, Bte, Cte, limiares_ts, "TESTE_TS", pasta)
        extras["T"] = np.array([res["T"]])

    # logits brutos + alvos + nome do exame: permitem bootstrap, IC por classe e testes pareados
    nomes = lambda p: np.array([os.path.basename(f) for f in particoes[p]])
    np.savez(os.path.join(pasta, "logits.npz"), Lvl=Lvl, Bvl=Bvl, Cvl=Cvl, Lte=Lte, Bte=Bte, Cte=Cte,
             arq_vl=nomes("val"), arq_te=nomes("test"), **extras)
    res["tempo_s"] = round(time.time() - t0, 1)
    with open(os.path.join(pasta, "metricas.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n Seed {seed} concluida em {time.strftime('%H:%M:%S', time.gmtime(res['tempo_s']))}")
    return res


def agregar(resultados: list[dict], cfg: Config) -> dict:
    """Media, desvio padrao amostral (ddof=1) e semi-amplitude do IC95% (t de Student)
    das metricas macro de teste entre seeds."""
    partes = ["teste"] + (["teste_ts"] if cfg.apply_temperature_scaling else [])
    n = len(resultados)
    t_crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    linhas, resumo = [], {"n_seeds": n, "seeds": [r["seed"] for r in resultados]}
    for parte in partes:
        resumo[parte] = {}
        for k in METRICAS:
            v = np.array([r[parte][k] for r in resultados], float)
            dp = float(v.std(ddof=1)) if n > 1 else float("nan")
            resumo[parte][k] = {"media": float(v.mean()), "dp": dp, "ic95": t_crit * dp / np.sqrt(n),
                                "valores": v.tolist()}
        linhas += [{"seed": r["seed"], "parte": parte, **{k: r[parte][k] for k in METRICAS}}
                   for r in resultados]
    if cfg.apply_temperature_scaling:
        resumo["T"] = [r["T"] for r in resultados]

    pasta = dir_experimento(cfg)
    pd.DataFrame(linhas).to_csv(os.path.join(pasta, "resumo_por_seed.csv"), index=False)
    with open(os.path.join(pasta, "resumo.json"), "w") as f:
        json.dump(resumo, f, indent=2)

    print(f"\n{'='*64}\n RESUMO {cfg.loss_type} | {n} seeds {resumo['seeds']}\n{'='*64}")
    for parte in partes:
        print(f"\n[{parte.upper()}]  media +- dp  (IC95 +-)")
        for k, nome in METRICAS.items():
            m = resumo[parte][k]
            print(f"  {nome:<7}{m['media']:>9.4f} +- {m['dp']:.4f}  (+- {m['ic95']:.4f})")
    return resumo


def treinar(cfg: Config) -> dict:
    t0 = time.time()
    pasta = dir_experimento(cfg)
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, "config.json"), "w") as f:
        json.dump({k: (str(v) if k == "device" else v) for k, v in dataclasses.asdict(cfg).items()}, f, indent=2)
    print(f"Dispositivo: {cfg.device}")

    particoes = listar_particoes(cfg)  # lido uma vez, reutilizado por todas as seeds
    with open(os.path.join(pasta, "particoes.json"), "w") as f:
        json.dump(resumir_particoes(particoes, cfg), f, indent=2)

    resultados = [rodar_seed(cfg, particoes, s, mostrar_sumario=(i == 0)) for i, s in enumerate(cfg.seeds)]
    resumo = agregar(resultados, cfg)
    print(f"\n Tempo total: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")
    return resumo


@torch.no_grad()
def inferir(cfg: Config, seed: int, n: int = 5, particao: str = "val") -> None:
    """Mostra n exemplos de uma particao usando o modelo treinado de uma seed."""
    pasta = os.path.join(dir_experimento(cfg), f"seed_{seed}")
    modelo = construir_modelo(cfg)
    for m, arq in zip(_redes(modelo), arquivos_pesos(cfg, pasta)):
        m.load_state_dict(torch.load(arq, map_location=cfg.device))
        m.to(cfg.device).eval()
    arq_t = os.path.join(pasta, "temperatura.npy")
    usar_ts = cfg.apply_temperature_scaling and os.path.exists(arq_t)
    T = float(np.load(arq_t)[0]) if usar_ts else 1.0
    limiares = np.load(os.path.join(pasta, "limiares_ts.npy" if usar_ts else "limiares.npy"))
    arqs = listar_particoes(cfg)[particao]

    rng = np.random.default_rng(seed)
    ds = ECGDataset(arqs)
    print(f"seed {seed} | particao {particao} | T = {T:.3f}")
    for k, i in enumerate(int(j) for j in rng.choice(len(ds), n, replace=False)):
        x, c = ds[i]
        probs = torch.sigmoid(forward_logits(modelo, x.unsqueeze(0).to(cfg.device)) / T).squeeze(0).cpu().numpy()
        pred, real = probs >= limiares, (c >= cfg.limiar_presenca).numpy()
        print(f"\n[{k+1}/{n}] {os.path.basename(arqs[i])}")
        print("  confiabilidade:  " + " | ".join(f"{CLASSES[j]}:{probs[j]:.2f}" for j in range(len(CLASSES))))
        print("  presentes (real):", ", ".join(CLASSES[j] for j in range(len(CLASSES)) if real[j]) or "nenhuma")
        print("  detectadas:      ", ", ".join(CLASSES[j] for j in range(len(CLASSES)) if pred[j]) or "nenhuma")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["treinar", "inferir"], nargs="?", default="treinar")
    ap.add_argument("--arch", choices=ARCH_TYPES, help="sobrescreve cfg.arch_type (shared=B, specialists=A)")
    ap.add_argument("--loss", choices=LOSS_TYPES, help="sobrescreve cfg.loss_type")
    ap.add_argument("--seeds", type=int, nargs="+", help="sobrescreve cfg.seeds")
    ap.add_argument("--ts", action="store_true", help="liga cfg.apply_temperature_scaling")
    ap.add_argument("--caminho", help="sobrescreve cfg.caminho (pasta com train/ val/ test/)")
    ap.add_argument("--out", help="sobrescreve cfg.out_dir")
    ap.add_argument("--seed", type=int, help="inferir: seed do modelo treinado (padrao: a primeira)")
    ap.add_argument("--n", type=int, default=5, help="inferir: numero de exemplos")
    ap.add_argument("--particao", choices=["val", "test"], default="val", help="inferir: particao")
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    sobrescritas = {k: v for k, v in {"arch_type": a.arch, "loss_type": a.loss, "seeds": a.seeds,
                                      "caminho": a.caminho,
                                      "out_dir": a.out}.items() if v is not None}
    if a.ts:
        sobrescritas["apply_temperature_scaling"] = True
    cfg = Config(**sobrescritas)
    if a.cmd == "treinar":
        treinar(cfg)
    else:
        inferir(cfg, a.seed if a.seed is not None else cfg.seeds[0], a.n, a.particao)