"""Arquitetura (backbone compartilhado SE+FFT + 5 saidas) e funcoes de perda.

As classes BandPass, SEBlock, ResBlock1D, SelfAttention1D e SharedBackboneNet sao
mantidas sem alteracao em relacao a versao do artigo (mesmos nomes de modulos,
logo os state_dict antigos continuam carregaveis). Apenas SharedLoss mudou."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from config import LOSS_TYPES

try:
    from torchaudio.functional import highpass_biquad, lowpass_biquad
    _TA = True
except Exception:
    _TA = False
    import warnings
    warnings.warn("torchaudio not found: the 0.5-120 Hz band-pass filter is DISABLED "
                  "(results will not match the paper). Install torchaudio.", stacklevel=2)


class BandPass(nn.Module):
    """Filtro passa-banda 0.5-120 Hz (no-op se torchaudio ausente)."""
    def __init__(self, sr=500, lo=0.5, hi=120.0):
        super().__init__()
        self.sr, self.lo, self.hi = sr, lo, hi

    def forward(self, x):
        if not _TA:
            return x
        return lowpass_biquad(highpass_biquad(x, self.sr, self.lo), self.sr, self.hi)


class SEBlock(nn.Module):
    """Squeeze-and-excitation: recalibra canais pela media temporal."""
    def __init__(self, c, reduction=8):
        super().__init__()
        h = max(c // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True),
                                nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ResBlock1D(nn.Module):
    """Bloco residual 1D pre-ativado com convolucoes dilatadas e SE."""
    def __init__(self, c_in, c_out, dilation=1, kernel=3, drop=0.2, use_se=True):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.bn1 = nn.BatchNorm1d(c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(drop)
        self.se = SEBlock(c_out) if use_se else nn.Identity()
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(self.act(self.bn2(out)))
        return self.se(out) + self.skip(x)


class SelfAttention1D(nn.Module):
    """Autoatencao temporal com conexao residual ponderada (gamma inicia em 0)."""
    def __init__(self, c, drop=None):
        super().__init__()
        self.Q, self.K, self.V = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1))
        self.drop = drop

    def forward(self, x):
        xt = x.transpose(-1, -2)
        att = F.softmax(self.Q(xt) @ self.K(xt).transpose(-1, -2) / (xt.size(-1) ** 0.5), -1)
        if self.training and self.drop is not None:
            att = F.dropout(att, self.drop)
        return x + self.gamma * (att @ self.V(xt)).transpose(-1, -2)


def _stage(c_in, filtros, blocos, drop, use_se):
    seq, c = [], c_in
    for f, n in zip(filtros, blocos):
        for i in range(n):
            seq.append(ResBlock1D(c, f, dilation=2 ** min(i, 4), drop=drop, use_se=use_se))
            c = f
        seq.append(nn.MaxPool1d(2))
    return nn.Sequential(*seq), c


def _init_pesos(modulo):
    for m in modulo.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _ramos(n_leads, stage_filters, stage_blocks, spec_filters, spec_blocks, drop, use_se, usar_fft):
    """Constroi o ramo temporal e, se usar_fft, o ramo espectral. Retorna os
    modulos (feat_s/stem_s = None se sem FFT) e a dimensao concatenada apos pooling."""
    stem_t = nn.Sequential(nn.BatchNorm1d(n_leads),
                           nn.Conv1d(n_leads, stage_filters[0], 7, stride=2, padding=3),
                           nn.BatchNorm1d(stage_filters[0]), nn.ReLU(inplace=True))
    feat_t, c_t = _stage(stage_filters[0], stage_filters, stage_blocks, drop, use_se)
    if not usar_fft:
        return stem_t, feat_t, c_t, None, None, 0
    stem_s = nn.Sequential(nn.BatchNorm1d(n_leads),
                           nn.Conv1d(n_leads, spec_filters[0], 7, stride=2, padding=3),
                           nn.BatchNorm1d(spec_filters[0]), nn.ReLU(inplace=True))
    feat_s, c_s = _stage(spec_filters[0], spec_filters, spec_blocks, drop, use_se)
    return stem_t, feat_t, c_t, stem_s, feat_s, c_s


def _forward_ramos(self, x):
    """Forward do ramo temporal (+ espectral se houver). A atencao e o ramo
    espectral sao aplicados conforme a configuracao de ablacao do modelo."""
    x = self.bandpass(x)
    t = self.feat_t(self.stem_t(x))
    if self.attn_t is not None:
        t = self.attn_t(t)
    t = torch.cat([t.mean(-1), t.amax(-1)], dim=1)
    if self.feat_s is None:                      # sem ramo espectral (ablacao)
        return self.feat_drop(t)
    janela = torch.hann_window(x.size(-1), device=x.device, dtype=x.dtype)
    xf = torch.log1p(torch.fft.rfft(x * janela, dim=-1).abs())
    s = self.feat_s(self.stem_s(xf))
    s = torch.cat([s.mean(-1), s.amax(-1)], dim=1)
    return self.feat_drop(torch.cat([t, s], dim=1))


def print_summary(modelo, dataloader, device, depth=2, max_lines=30):
    """Sumario via torchinfo, limpando hooks que poderiam afetar treinos seguintes."""
    try:
        x, _ = next(iter(dataloader))
        info = summary(modelo.to(device), input_data=x.to(device),
                       col_names=("input_size", "output_size", "num_params", "trainable"),
                       depth=depth, verbose=0)
    finally:
        for m in modelo.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()
        modelo.zero_grad(set_to_none=True)
    linhas = str(info).splitlines()
    if max_lines and len(linhas) > max_lines:
        k = max_lines // 2
        linhas = linhas[:k] + ["..."] + linhas[-k:]
    print("\n".join(linhas))


class SharedBackboneNet(nn.Module):
    """Backbone unico (mesmos ramos temporal+espectral) com 5 cabecas de saida,
    uma por superclasse. Compartilha caracteristicas entre patologias por
    construcao; saida: 5 logits. FFT, SE e autoatencao sao opcionais (ablacao)."""
    def __init__(self, n_leads=12, n_classes=5, stage_filters=(16, 32, 64),
                 stage_blocks=(3, 5, 7), spec_filters=(16, 32), spec_blocks=(2, 2),
                 drop=0.15, usar_se=True, usar_fft=True, usar_attn=True):
        super().__init__()
        self.bandpass = BandPass()
        (self.stem_t, self.feat_t, c_t,
         self.stem_s, self.feat_s, c_s) = _ramos(n_leads, stage_filters, stage_blocks,
                                                 spec_filters, spec_blocks, drop, usar_se, usar_fft)
        self.attn_t = SelfAttention1D(c_t, drop) if usar_attn else None
        self.feat_drop = nn.Dropout(drop)
        self.cabeca = nn.Sequential(
            nn.Linear(2 * (c_t + c_s), 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_classes))
        _init_pesos(self)

    def forward(self, x):
        return self.cabeca(_forward_ramos(self, x))


class SpecialistResNet(nn.Module):
    """Rede especialista de uma unica superclasse (Modelo A); saida: 1 logit."""
    def __init__(self, n_leads=12, stage_filters=(8, 16, 32), stage_blocks=(3, 5, 7),
                 spec_filters=(8, 16), spec_blocks=(2, 2), drop=0.15,
                 usar_se=True, usar_fft=True, usar_attn=False):
        super().__init__()
        self.bandpass = BandPass()
        (self.stem_t, self.feat_t, c_t,
         self.stem_s, self.feat_s, c_s) = _ramos(n_leads, stage_filters, stage_blocks,
                                                 spec_filters, spec_blocks, drop, usar_se, usar_fft)
        self.attn_t = SelfAttention1D(c_t, drop) if usar_attn else None
        self.feat_drop = nn.Dropout(drop)
        self.cabeca = nn.Sequential(
            nn.Linear(2 * (c_t + c_s), 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1))
        _init_pesos(self)

    def forward(self, x):
        return self.cabeca(_forward_ramos(self, x))


class SharedLoss(nn.Module):
    """Perda multirrotulo das 5 saidas, selecionada por loss_type.

    bce_soft        : BCE com alvo = confiabilidade continua em [0,1] (proposta do artigo).
    bce_hard        : BCE com alvo binario (c >= limiar). Baseline do artigo.
    label_smoothing : alvo binario suavizado, 0 -> eps/2 e 1 -> 1 - eps/2, depois BCE.
    focal_loss      : FL = -alpha_t (1 - p_t)^gamma log(p_t), elemento a elemento, sobre
                      o alvo binario; reduz o peso de exemplos ja bem classificados.

    pos_weight e aplicado no termo de log-verossimilhanca de TODOS os regimes, para
    que as comparacoes nao misturem o efeito do alvo com o do balanceamento de HYP.
    Na focal, alpha (opcional) se soma a isso; com gamma=0 e alpha=None ela e
    identica a bce_hard, e label_smoothing com eps=0 tambem."""
    def __init__(self, pos_weight: Sequence[float] | None = None, loss_type: str = "bce_soft",
                 limiar: float = 0.5, ls_epsilon: float = 0.1, focal_gamma: float = 2.0,
                 focal_alpha: float | None = None):
        super().__init__()
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type desconhecido: {loss_type!r}")
        self.loss_type, self.limiar = loss_type, limiar
        self.eps, self.gamma, self.alpha = ls_epsilon, focal_gamma, focal_alpha
        pw = torch.ones(5) if pos_weight is None else pos_weight
        self.register_buffer("pw", torch.as_tensor(pw, dtype=torch.float32))

    def alvo(self, targets: torch.Tensor) -> torch.Tensor:
        """Converte a confiabilidade [0,1] no alvo efetivo do regime escolhido."""
        t = targets.float()
        if self.loss_type == "bce_soft":
            return t
        t = (t >= self.limiar).float()
        if self.loss_type == "label_smoothing":
            t = t * (1.0 - self.eps) + self.eps / 2.0
        return t

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = self.alvo(targets)
        if self.loss_type != "focal_loss":
            return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pw)
        bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * t + (1.0 - p) * (1.0 - t)
        perda = (1.0 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            perda = (self.alpha * t + (1.0 - self.alpha) * (1.0 - t)) * perda
        return perda.mean()


class SpecialistLoss(nn.Module):
    """Perda especialista para a superclasse de indice `specialty_idx`.
    
    Funciona de forma idêntica à SharedLoss (suporta bce_soft, bce_hard, 
    label_smoothing e focal_loss), mas isola o alvo e a predição para uma 
    única patologia. Aplica o pos_weight específico da classe."""
    def __init__(self, specialty_idx: int, pos_weight: float = 1.0, loss_type: str = "bce_soft",
                 limiar: float = 0.5, ls_epsilon: float = 0.1, focal_gamma: float = 2.0,
                 focal_alpha: float | None = None):
        super().__init__()
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type desconhecido: {loss_type!r}")
        self.specialty_idx = specialty_idx
        self.loss_type, self.limiar = loss_type, limiar
        self.eps, self.gamma, self.alpha = ls_epsilon, focal_gamma, focal_alpha
        self.register_buffer("pw", torch.tensor(float(pos_weight), dtype=torch.float32))

    def alvo(self, targets: torch.Tensor) -> torch.Tensor:
        """Isola o alvo da superclasse atual e converte para o regime escolhido."""
        t = targets[:, self.specialty_idx].float()
        if self.loss_type == "bce_soft":
            return t
        t = (t >= self.limiar).float()
        if self.loss_type == "label_smoothing":
            t = t * (1.0 - self.eps) + self.eps / 2.0
        return t

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = self.alvo(targets)
        z = logits.squeeze(1)  # Reduz a predição de (N, 1) para (N,) para casar com o alvo
        
        if self.loss_type != "focal_loss":
            return F.binary_cross_entropy_with_logits(z, t, pos_weight=self.pw)
        
        bce = F.binary_cross_entropy_with_logits(z, t, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(z)
        p_t = p * t + (1.0 - p) * (1.0 - t)
        perda = (1.0 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            perda = (self.alpha * t + (1.0 - self.alpha) * (1.0 - t)) * perda
        return perda.mean()