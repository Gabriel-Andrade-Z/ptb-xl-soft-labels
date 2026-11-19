"""Treino, avaliacao e pos-processamento (temperature scaling).

As funcoes de avaliacao (otimizar_limiares, avaliar_multilabel, avaliar_regressao,
avaliar_calibracao, plot_confusao_multilabel) e o laco de treino sao os mesmos da
versao do artigo; a unica adicao e TemperatureScaler."""
from __future__ import annotations

import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             mean_absolute_error, mean_squared_error, confusion_matrix)

from config import CLASSES


class EarlyStopping:
    """Early stop para val_loss (quanto menor, melhor): salva o melhor estado pela
    loss REAL (best_raw) e decide a parada pela loss SUAVIZADA (best_smooth),
    reduzindo o efeito da oscilacao no criterio de parada."""
    def __init__(self, patience, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_raw = math.inf
        self.best_smooth = math.inf
        self.estado, self.epoca, self.cont = None, 0, 0

    def passo(self, raw_loss, smooth_loss, modelo, e):
        if raw_loss < self.best_raw - self.min_delta:
            self.best_raw = raw_loss
            self.estado = copy.deepcopy(modelo.state_dict())
            self.epoca = e
        if smooth_loss < self.best_smooth - self.min_delta:
            self.best_smooth = smooth_loss
            self.cont = 0
        else:
            self.cont += 1
        return self.cont >= self.patience

    def restaurar(self, modelo):
        if self.estado is not None:
            modelo.load_state_dict(self.estado)



def cosine_warmup(opt, warmup, total):
    """Scheduler: warmup linear seguido de decaimento cosseno."""
    def fn(e):
        if e < warmup:
            return (e + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min((e - warmup) / max(1, total - warmup), 1.0)))
    return optim.lr_scheduler.LambdaLR(opt, fn)


# ====================== AVALIACAO MULTIRROTULO (comum a A e B) ======================

def otimizar_limiares(probs, alvos):
    """Para cada classe, escolhe na grade [0.05, 0.95] o limiar que maximiza o F1,
    usando SOMENTE o conjunto passado (a validacao). Aplicado fixo ao teste cego."""
    probs, alvos = np.asarray(probs, float), np.asarray(alvos, float)
    limiares = np.full(len(CLASSES), 0.5)
    for k in range(len(CLASSES)):
        if alvos[:, k].sum() == 0:
            continue
        f1s = [f1_score(alvos[:, k], (probs[:, k] > t).astype(int), zero_division=0)
               for t in np.linspace(0.05, 0.95, 91)]
        limiares[k] = np.linspace(0.05, 0.95, 91)[int(np.argmax(f1s))]
    return limiares


def avaliar_multilabel(probs, alvos, titulo, caminho_prefixo, limiares=None):
    """AUC-ROC e AUPRC (independentes de limiar) e F1 (em 0.5 e nos limiares
    otimizados), por classe e macro. Gera grafico de barras por classe."""
    probs, alvos = np.asarray(probs, float), np.asarray(alvos, float)
    aucs, aps, f1_05, f1_opt = [], [], [], []
    print(f"\n{'='*64}\n {titulo}\n{'='*64}")
    cab = f"{'classe':<8}{'AUC-ROC':>9}{'AUPRC':>9}{'F1@0.5':>9}"
    if limiares is not None:
        cab += f"{'F1*':>9}{'t*':>7}"
    print(cab + f"{'real+':>8}")

    for k, nome in enumerate(CLASSES):
        yk, pk = alvos[:, k], probs[:, k]
        pos = int(yk.sum())
        auc = roc_auc_score(yk, pk) if 0 < pos < len(yk) else float("nan")
        ap = average_precision_score(yk, pk) if 0 < pos < len(yk) else float("nan")
        f1m = f1_score(yk, (pk > 0.5).astype(int), zero_division=0)
        aucs.append(auc); aps.append(ap); f1_05.append(f1m)
        linha = f"{nome:<8}{auc:>9.4f}{ap:>9.4f}{f1m:>9.4f}"
        if limiares is not None:
            fo = f1_score(yk, (pk > limiares[k]).astype(int), zero_division=0)
            f1_opt.append(fo)
            linha += f"{fo:>9.4f}{limiares[k]:>7.2f}"
        print(linha + f"{pos:>8}")

    auc_macro, ap_macro, f1_macro = np.nanmean(aucs), np.nanmean(aps), np.mean(f1_05)
    print("-" * 60)
    rod = f"{'MACRO':<8}{auc_macro:>9.4f}{ap_macro:>9.4f}{f1_macro:>9.4f}"
    if limiares is not None:
        rod += f"{np.mean(f1_opt):>9.4f}"
    print(rod)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cores = plt.cm.magma(np.linspace(0.2, 0.8, len(CLASSES)))
    for a, vals, nome, lo in [(ax[0], aucs, "AUC-ROC", 0.5), (ax[1], aps, "AUPRC", 0.0)]:
        a.bar(CLASSES, vals, color=cores); a.set_title(f"{titulo} — {nome}")
        a.set_ylim(lo, 1.0); a.grid(True, axis="y", alpha=0.3)
        a.axhline(np.nanmean(vals), ls="--", c="k", alpha=0.6, label=f"macro {np.nanmean(vals):.3f}")
        a.legend()
    os.makedirs(os.path.dirname(caminho_prefixo), exist_ok=True)
    plt.tight_layout(); plt.savefig(f"{caminho_prefixo}_auc.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    res = {"auc_macro": float(auc_macro), "ap_macro": float(ap_macro), "f1_macro_05": float(f1_macro)}
    if limiares is not None:
        res["f1_macro_opt"] = float(np.mean(f1_opt))
    return res


def plot_confusao_multilabel(probs, alvos, limiares, titulo, caminho):
    """Cinco matrizes de confusao 2x2 (uma por patologia), anotadas com contagem,
    recall e precisao por celula, no limiar otimizado de cada classe."""
    probs, alvos = np.asarray(probs, float), np.asarray(alvos, float)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 4))
    for k, (nome, ax) in enumerate(zip(CLASSES, axes)):
        cm = confusion_matrix(alvos[:, k].astype(int), (probs[:, k] > limiares[k]).astype(int), labels=[0, 1])
        rec = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros(cm.shape), where=cm.sum(1, keepdims=True) != 0) * 100
        pre = np.divide(cm, cm.sum(0, keepdims=True), out=np.zeros(cm.shape), where=cm.sum(0, keepdims=True) != 0) * 100
        anot = np.array([[f"{cm[i,j]}\nR:{rec[i,j]:.0f}%\nP:{pre[i,j]:.0f}%" for j in range(2)] for i in range(2)])
        sns.heatmap(rec, annot=anot, fmt="", cmap="magma", vmin=0, vmax=100, cbar=False, ax=ax,
                    xticklabels=["ausente", "presente"], yticklabels=["ausente", "presente"])
        ax.set_title(f"{nome} (t*={limiares[k]:.2f})")
        ax.set_xlabel("Predito"); ax.set_ylabel("Real" if k == 0 else "")
    fig.suptitle(titulo); plt.tight_layout()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=130, bbox_inches="tight"); plt.close(fig)


def avaliar_regressao(probs, alvos_cont, titulo):
    """Qualidade da estimativa de confianca (soft labels): MAE, RMSE e MSE por
    classe + macro, e MAE por faixa de confiabilidade real (mede se a rede
    distingue suspeita fraca/moderada/alta ou apenas 0 vs nao-zero)."""
    probs, alvos_cont = np.asarray(probs, float), np.asarray(alvos_cont, float)
    print(f"\n{'='*60}\n {titulo} — Estimativa de confianca (regressao)\n{'='*60}")
    print(f"{'classe':<8}{'MAE':>10}{'RMSE':>10}{'MSE':>10}")
    maes, rmses = [], []
    for k, nome in enumerate(CLASSES):
        mae = mean_absolute_error(alvos_cont[:, k], probs[:, k])
        rmse = math.sqrt(mean_squared_error(alvos_cont[:, k], probs[:, k]))
        maes.append(mae); rmses.append(rmse)
        print(f"{nome:<8}{mae:>10.4f}{rmse:>10.4f}{rmse**2:>10.4f}")
    print("-" * 38)
    print(f"{'MACRO':<8}{np.mean(maes):>10.4f}{np.mean(rmses):>10.4f}{np.mean([r**2 for r in rmses]):>10.4f}")

    print(f"\n MAE por faixa de confiabilidade real:")
    print(f"{'faixa':<10}{'n':>10}{'MAE':>10}")
    a, p = alvos_cont.ravel(), probs.ravel()
    for v in [0.0, 0.15, 0.35, 0.50, 0.80, 1.0]:
        m = np.isclose(a, v, atol=0.01)
        if m.sum():
            print(f"{v:<10.2f}{int(m.sum()):>10}{mean_absolute_error(a[m], p[m]):>10.4f}")
    return {"mae_macro": float(np.mean(maes)), "rmse_macro": float(np.mean(rmses))}


def avaliar_calibracao(probs, alvos_bin, titulo, caminho, n_bins=10):
    """Calibracao das confiabilidades preditas (revisao): ECE e Brier por classe
    + macro, e diagrama de confiabilidade (reliability diagram). Mede se a
    probabilidade predita corresponde a frequencia real de presenca."""
    probs, alvos_bin = np.asarray(probs, float), np.asarray(alvos_bin, float)
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n{'='*60}\n {titulo} — Calibracao (ECE, Brier)\n{'='*60}")
    print(f"{'classe':<8}{'ECE':>10}{'Brier':>10}")
    eces, briers, curvas = [], [], []
    for k, nome in enumerate(CLASSES):
        y, p = alvos_bin[:, k], probs[:, k]
        brier = float(np.mean((p - y) ** 2))
        ece, conf_bin, acc_bin = 0.0, [], []
        for i in range(n_bins):
            m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
            if m.sum() == 0:
                conf_bin.append(np.nan); acc_bin.append(np.nan); continue
            conf, acc = p[m].mean(), y[m].mean()
            conf_bin.append(conf); acc_bin.append(acc)
            ece += (m.sum() / len(p)) * abs(conf - acc)
        eces.append(ece); briers.append(brier); curvas.append((conf_bin, acc_bin))
        print(f"{nome:<8}{ece:>10.4f}{brier:>10.4f}")
    print("-" * 28)
    print(f"{'MACRO':<8}{np.mean(eces):>10.4f}{np.mean(briers):>10.4f}")

    centros = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(1, len(CLASSES), figsize=(3.2 * len(CLASSES), 3.2))
    for k, (nome, (cb, ab)) in enumerate(zip(CLASSES, curvas)):
        ax[k].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax[k].plot(centros, ab, "o-", color="#FF6B35", lw=1.5, ms=4)
        ax[k].set_title(f"{nome} (ECE={eces[k]:.3f})", fontsize=9)
        ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1)
        ax[k].set_xlabel("Confianca predita", fontsize=8)
        ax[k].set_ylabel("Freq. real" if k == 0 else "", fontsize=8)
        ax[k].grid(True, alpha=0.3)
    fig.suptitle(titulo); plt.tight_layout()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"ece_macro": float(np.mean(eces)), "brier_macro": float(np.mean(briers))}


# ====================== CALIBRACAO POS-TREINO (TEMPERATURE SCALING) ======================

class TemperatureScaler(nn.Module):
    """Temperature scaling (Guo et al., 2017) adaptado ao caso multirrotulo: um unico
    T escalar compartilhado pelas 5 saidas, p = sigmoid(logits / T).

    T e ajustado por L-BFGS minimizando a BCE (sem pos_weight) entre os logits de
    VALIDACAO e o alvo BINARIO de presenca, que e o mesmo alvo usado por ECE/Brier.
    Por ser uma transformacao monotona por classe, AUC e AUPRC nao mudam; os
    limiares de decisao, porem, precisam ser reotimizados sobre as probs escaladas."""
    def __init__(self, t_init: float = 1.5):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(float(t_init)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def ajustar(self, logits_val: np.ndarray, alvos_bin_val: np.ndarray, max_iter: int = 100) -> float:
        z = torch.as_tensor(logits_val, dtype=torch.float32)
        y = torch.as_tensor(alvos_bin_val, dtype=torch.float32)
        opt = optim.LBFGS([self.T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            perda = F.binary_cross_entropy_with_logits(self(z), y)
            perda.backward()
            return perda

        with torch.no_grad():
            nll0 = F.binary_cross_entropy_with_logits(z, y).item()
        opt.step(closure)
        with torch.no_grad():
            nll1 = F.binary_cross_entropy_with_logits(self(z), y).item()
        print(f"Temperature scaling: T = {self.temperatura:.4f} | NLL val {nll0:.4f} -> {nll1:.4f}")
        return self.temperatura

    @property
    def temperatura(self) -> float:
        return float(self.T.detach())

    @torch.no_grad()
    def escalar(self, logits: np.ndarray) -> np.ndarray:
        """Retorna logits / T (para aplicar ao teste)."""
        return self(torch.as_tensor(logits, dtype=torch.float32)).numpy()


# ====================== TREINO DO ESPECIALISTA (Modelo A) ======================

@torch.no_grad()
def _avaliar_esp(modelo, ld, crit, cfg):
    modelo.eval()
    soma, n = 0.0, 0
    for x, c in ld:
        x, c = x.to(cfg.device), c.to(cfg.device)
        soma += crit(modelo(x), c).item() * x.size(0); n += x.size(0)
    return soma / n


def treinar_especialista(modelo, ld_tr, ld_vl, crit, cfg, tag="Especialista", lr=None):
    """Treina uma rede especialista com a perda escolhida, early stop por val_loss
    suavizado (media movel) para reduzir o efeito da oscilacao."""
    modelo.to(cfg.device)
    if lr is None:
        lr = cfg.lr_cnn
    opt = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sch = cosine_warmup(opt, cfg.warmup, cfg.epochs_cnn)
    parada = EarlyStopping(cfg.es_patience)
    vll_suave = None
    print(f"\n>> {tag} (loss={cfg.loss_type}, early stop por val_loss suavizado)")
    for e in range(1, cfg.epochs_cnn + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, c in ld_tr:
            x, c = x.to(cfg.device), c.to(cfg.device)
            opt.zero_grad()
            loss = crit(modelo(x), c)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            opt.step()
            soma += loss.item() * x.size(0); n += x.size(0)
        vll = _avaliar_esp(modelo, ld_vl, crit, cfg)
        sch.step()
        vll_suave = vll if vll_suave is None else cfg.es_suavizacao * vll_suave + (1 - cfg.es_suavizacao) * vll
        print(f"  E{e:03d} | loss {soma/n:.4f} val_loss {vll:.4f} | lr {opt.param_groups[0]['lr']:.2e}")
        if parada.passo(vll, vll_suave, modelo, e):
            print(f"  Early stop @e{e} (melhor val_loss {parada.best_raw:.4f} @e{parada.epoca})")
            break
    parada.restaurar(modelo)
    return modelo

# ====================== TREINO DO BACKBONE COMPARTILHADO (Modelo B) ======================

@torch.no_grad()
def _avaliar_shared(modelo, ld, crit, cfg):
    modelo.eval()
    soma, n = 0.0, 0
    for x, c in ld:
        x, c = x.to(cfg.device), c.to(cfg.device)
        soma += crit(modelo(x), c).item() * x.size(0); n += x.size(0)
    return soma / n


def treinar_compartilhado(modelo, ld_tr, ld_vl, crit, cfg, tag="ModeloB"):
    """Treina o backbone unico com a perda multirrotulo `crit`, early stop por val_loss
    suavizado (media movel) para reduzir o efeito da oscilacao."""
    modelo.to(cfg.device)
    opt = optim.AdamW(modelo.parameters(), lr=cfg.lr_cnn, weight_decay=cfg.weight_decay)
    sch = cosine_warmup(opt, cfg.warmup, cfg.epochs_cnn)
    parada = EarlyStopping(cfg.es_patience)
    vll_suave = None
    print(f"\n>> {tag} (loss={cfg.loss_type}, early stop por val_loss suavizado)")
    for e in range(1, cfg.epochs_cnn + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, c in ld_tr:
            x, c = x.to(cfg.device), c.to(cfg.device)
            opt.zero_grad()
            loss = crit(modelo(x), c)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            opt.step()
            soma += loss.item() * x.size(0); n += x.size(0)
        vll = _avaliar_shared(modelo, ld_vl, crit, cfg)
        sch.step()
        vll_suave = vll if vll_suave is None else cfg.es_suavizacao * vll_suave + (1 - cfg.es_suavizacao) * vll
        print(f"  E{e:03d} | loss {soma/n:.4f} val_loss {vll:.4f} | lr {opt.param_groups[0]['lr']:.2e}")
        if parada.passo(vll, vll_suave, modelo, e):
            print(f"  Early stop @e{e} (melhor val_loss {parada.best_raw:.4f} @e{parada.epoca})")
            break
    parada.restaurar(modelo)
    return modelo


