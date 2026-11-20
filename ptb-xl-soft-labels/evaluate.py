"""Recompute every number reported in the paper from the per-seed test logits.

Usage:
    python evaluate.py                      # reads results/, writes results/stats.json
    python evaluate.py --results runs_dir --out my_stats.json --jobs 4

Expected layout (the same one written by train.py):
    <results>/runs/shared_<loss>/seed_<s>/{logits.npz, metricas.json}
    <results>/runs/shared_bce_soft/particoes.json
    <results>/ablation/<A|B>_<variant>/metricas.json      (Table III, validation)

Recalcula todas as metricas do artigo a partir dos logits.npz das 5 seeds.

Formulas identicas a engine.py (AUC/AUPRC sklearn, F1* com limiares do val em
grade 0.05:0.01:0.95, ECE 10 bins, Brier vs alvo binario, MAE/RMSE vs confiabilidade
continua, MAE por faixa com atol=0.01). Adiciona: temperature scaling post-hoc em
todos os regimes, testes pareados (seed e bootstrap por exame) e incerteza por faixa.
Saida: stats.json (fonte unica dos numeros do manuscrito)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ap = argparse.ArgumentParser(description="Recompute results/stats.json from the saved logits.")
ap.add_argument("--results", default="results", help="folder with runs/ and ablation/")
ap.add_argument("--out", default=None, help="output JSON (default: <results>/stats.json)")
ap.add_argument("--n-boot", type=int, default=2000, help="paired bootstrap resamples (paper: 2000)")
ap.add_argument("--jobs", type=int, default=2, help="parallel processes for the bootstrap")
ARGS = ap.parse_args()
R = Path(ARGS.results)
OUT = Path(ARGS.out) if ARGS.out else R / "stats.json"
CLASSES = ["MI", "STTC", "CD", "HYP", "NORM"]
SEEDS = [42, 43, 44, 45, 46]
REG = {"bce_soft": "shared_bce_soft", "bce_hard": "shared_bce_hard",
       "label_smoothing": "shared_label_smoothing", "focal_loss": "shared_focal_loss"}
FAIXAS = [0.0, 0.15, 0.35, 0.50, 0.80, 1.0]
GRADE = np.linspace(0.05, 0.95, 91)
sig = lambda z: 1 / (1 + np.exp(-z))


def limiares(P, B):
    t = np.full(5, .5)
    for k in range(5):
        if B[:, k].sum():
            t[k] = GRADE[int(np.argmax([f1_score(B[:, k], (P[:, k] > g).astype(int), zero_division=0) for g in GRADE]))]
    return t


def ece_k(p, y, n=10):
    b = np.linspace(0, 1, n + 1); e = 0.0
    for i in range(n):
        m = (p >= b[i]) & ((p < b[i + 1]) if i < n - 1 else (p <= b[i + 1]))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def metricas(P, B, C, t):
    pc = {"auc": [roc_auc_score(B[:, k], P[:, k]) for k in range(5)],
          "ap": [average_precision_score(B[:, k], P[:, k]) for k in range(5)],
          "f1": [f1_score(B[:, k], (P[:, k] > t[k]).astype(int), zero_division=0) for k in range(5)],
          "ece": [ece_k(P[:, k], B[:, k]) for k in range(5)],
          "brier": [float(np.mean((P[:, k] - B[:, k]) ** 2)) for k in range(5)],
          "mae": [float(np.mean(np.abs(P[:, k] - C[:, k]))) for k in range(5)],
          "rmse": [float(np.sqrt(np.mean((P[:, k] - C[:, k]) ** 2))) for k in range(5)]}
    a, p = C.ravel(), P.ravel()
    faixa = {}
    for v in FAIXAS:
        m = np.isclose(a, v, atol=0.01)
        q = np.clip(p[m], 1e-7, 1 - 1e-7)
        faixa[str(v)] = {"n": int(m.sum()), "mae": float(np.mean(np.abs(a[m] - p[m]))),
                         "p_medio": float(p[m].mean()),
                         "entropia": float(np.mean(-(q * np.log2(q) + (1 - q) * np.log2(1 - q))))}
    macro = {k: float(np.mean(v)) for k, v in pc.items()}
    return {"macro": macro, "classe": pc, "faixa": faixa}


def ajustar_T(Lv, Bv):
    T = torch.nn.Parameter(torch.tensor(1.5))
    z, y = torch.tensor(Lv, dtype=torch.float32), torch.tensor(Bv, dtype=torch.float32)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(z / T, y); l.backward(); return l
    opt.step(cl)
    return float(T.detach())


# ------------------------------------------------------------------ fase 2
dados, res = {}, {}
for reg, sub in REG.items():
    for ts in (False, True):
        nome = reg + ("+TS" if ts else "")
        res[nome] = {"seeds": [], "T": []}
        for s in SEEDS:
            z = np.load(R / "runs" / sub / f"seed_{s}" / "logits.npz")
            Lv, Lt = z["Lvl"].astype(np.float64), z["Lte"].astype(np.float64)
            T = ajustar_T(Lv, z["Bvl"]) if ts else 1.0
            Pv, Pt = sig(Lv / T), sig(Lt / T)
            t = limiares(Pv, z["Bvl"])
            m = metricas(Pt, z["Bte"], z["Cte"], t)
            res[nome]["seeds"].append(m); res[nome]["T"].append(T)
            dados[(nome, s)] = (Pt, z["Bte"], z["Cte"], t)
            if not ts:  # confere contra o metricas.json salvo pelo pipeline
                ref = json.load(open(R / "runs" / sub / f"seed_{s}" / "metricas.json"))["teste"]
                for a, b in [("auc", "auc_macro"), ("ap", "ap_macro"), ("f1", "f1_macro_opt"), ("ece", "ece_macro"),
                             ("brier", "brier_macro"), ("mae", "mae_macro"), ("rmse", "rmse_macro")]:
                    assert abs(m["macro"][a] - ref[b]) < 1e-4, (nome, s, a, m["macro"][a], ref[b])
            elif reg == "bce_soft":
                ref = json.load(open(R / "runs" / sub / f"seed_{s}" / "metricas.json"))
                assert abs(T - ref["T"]) < 2e-3 and abs(m["macro"]["ece"] - ref["teste_ts"]["ece_macro"]) < 2e-3, (s, T, ref["T"])
print("conferencia com metricas.json: OK")

tc = stats.t.ppf(.975, len(SEEDS) - 1)
def agg(v):
    v = np.asarray(v, float); dp = v.std(ddof=1)
    return {"media": float(v.mean()), "dp": float(dp), "ic95": float(tc * dp / np.sqrt(len(v))), "valores": v.tolist()}

resumo = {}
for nome, r in res.items():
    S = r["seeds"]
    resumo[nome] = {"T": agg(r["T"]),
                    "macro": {k: agg([s["macro"][k] for s in S]) for k in S[0]["macro"]},
                    "classe": {k: {c: agg([s["classe"][k][i] for s in S]) for i, c in enumerate(CLASSES)} for k in S[0]["classe"]},
                    "faixa": {f: {q: agg([s["faixa"][f][q] for s in S]) for q in ("mae", "p_medio", "entropia")} | {"n": S[0]["faixa"][f]["n"]}
                              for f in S[0]["faixa"]}}

# ------------------------------------------------------------------ testes pareados
rng = np.random.default_rng(2026)
NB = ARGS.n_boot
N = dados[("bce_soft", 42)][0].shape[0]
idx_boot = rng.integers(0, N, (NB, N))

def macro_vec(P, B, C, t, met, I):
    """metrica macro para cada reamostragem I (NB,N); vetorizado quando possivel."""
    out = np.empty(len(I))
    for b, ii in enumerate(I):
        Pb, Bb, Cb = P[ii], B[ii], C[ii]
        if met == "auc":
            out[b] = np.mean([roc_auc_score(Bb[:, k], Pb[:, k]) for k in range(5)])
        elif met == "ap":
            out[b] = np.mean([average_precision_score(Bb[:, k], Pb[:, k]) for k in range(5)])
        elif met == "f1":
            out[b] = np.mean([f1_score(Bb[:, k], (Pb[:, k] > t[k]).astype(int), zero_division=0) for k in range(5)])
        elif met == "ece":
            out[b] = np.mean([ece_k(Pb[:, k], Bb[:, k]) for k in range(5)])
        elif met == "brier":
            out[b] = np.mean((Pb - Bb) ** 2)
        elif met == "mae":
            out[b] = np.mean(np.abs(Pb - Cb))
        elif met == "rmse":
            out[b] = np.mean(np.sqrt(np.mean((Pb - Cb) ** 2, 0)))
    return out

PARES = [("bce_soft", "bce_hard"), ("bce_soft", "bce_hard+TS"), ("bce_soft+TS", "bce_hard+TS"),
         ("bce_soft", "label_smoothing"), ("bce_soft", "focal_loss")]
METS = ["auc", "ap", "f1", "ece", "brier", "mae", "rmse"]
def _boot(tarefa):
    """Diferenca bootstrap (media sobre as seeds) para um par de regimes e uma metrica."""
    a, b, met = tarefa
    boot = np.zeros(NB)
    for s in SEEDS:
        boot += macro_vec(*dados[(a, s)], met, idx_boot) - macro_vec(*dados[(b, s)], met, idx_boot)
    return tarefa, boot / len(SEEDS)


tarefas = [(a, b, met) for a, b in PARES for met in METS]
try:  # 'fork' compartilha os logits com os processos filhos sem recarregar (Linux)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(ARGS.jobs) as pool:
        boots = dict(pool.map(_boot, tarefas))
except (ValueError, OSError):  # Windows/macOS sem fork: sequencial
    boots = dict(map(_boot, tarefas))

testes = {}
for a, b in PARES:
    chave = f"{a} vs {b}"; testes[chave] = {}
    for met in METS:
        va = np.array([s["macro"][met] for s in res[a]["seeds"]]); vb = np.array([s["macro"][met] for s in res[b]["seeds"]])
        d = va - vb
        tt = stats.ttest_rel(va, vb)
        w = stats.wilcoxon(va, vb, method="exact") if np.any(d != 0) else None
        boot = boots[(a, b, met)]
        p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        testes[chave][met] = {"delta_media": float(d.mean()), "delta_seeds": d.tolist(),
                              "t_p": float(tt.pvalue), "wilcoxon_p": float(w.pvalue) if w else 1.0,
                              "n_seeds_mesmo_sinal": int(max((d > 0).sum(), (d < 0).sum())),
                              "boot_ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                              "boot_p": p_boot}
    print("testes:", chave)

# ------------------------------------------------------------------ fase 1 (seed 42, validacao)
fase1 = {}
for arq in "AB":
    for var, cfg in [("base", "plain"), ("se", "se"), ("se_fft", "se_fft"), ("se_fft_attn", "se_fft_attn")]:
        m = json.load(open(R / "ablation" / f"{arq}_{var}" / "metricas.json"))
        fase1[f"{arq}_{cfg}"] = {"val": m["val"], "teste": m["teste"]}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump({"fase2": resumo, "testes": testes, "fase1": fase1,
           "particoes": json.load(open(R / "runs" / REG["bce_soft"] / "particoes.json"))},
          open(OUT, "w"), indent=1)
print("ok ->", OUT)
