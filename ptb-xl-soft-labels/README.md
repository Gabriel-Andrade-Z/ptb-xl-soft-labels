# Cardiologist Reliability-Based Soft Labels for Multi-Label ECG Classification

Code, patient-level split, training configurations and per-seed test logits for the paper
**"Improving Multi-Label ECG Classification Using Cardiologist Reliability-Based Soft Labels"**
(G. A. de Santana, V. de O. Soares, P. A. de A. da Silva, E. T. de Andrade, C. D. M. Regis — GPDS/IFPB).

PTB-XL provides, for every diagnostic statement, the reliability (0–100%) assigned by the annotating
cardiologist. Instead of discarding it when binary labels are created, we use it as a continuous
BCE target and compare it with hard labels, label smoothing, focal loss and post-hoc temperature
scaling under a **patient-level** partition, over **five seeds**.

## Main result (test set, mean ± 95% CI over seeds 42–46)

| Objective | AUC | AUPRC | F1* | ECE ↓ | Brier ↓ | MAE ↓ |
|---|---|---|---|---|---|---|
| Hard labels | 0.9202 ± 0.0031 | 0.7839 ± 0.0089 | 0.7159 ± 0.0085 | 0.0448 ± 0.0115 | 0.0874 ± 0.0036 | 0.1542 ± 0.0055 |
| Hard labels + TS | 0.9202 ± 0.0031 | 0.7839 ± 0.0089 | 0.7164 ± 0.0081 | 0.0428 ± 0.0088 | 0.0867 ± 0.0031 | 0.1655 ± 0.0063 |
| Label smoothing (ε = 0.1) | 0.9195 ± 0.0041 | 0.7840 ± 0.0103 | 0.7160 ± 0.0138 | 0.0737 ± 0.0122 | 0.0911 ± 0.0070 | 0.1988 ± 0.0122 |
| Focal loss (γ = 2) | 0.9178 ± 0.0040 | 0.7750 ± 0.0136 | 0.7056 ± 0.0071 | 0.1348 ± 0.0080 | 0.1063 ± 0.0042 | 0.2582 ± 0.0086 |
| **Soft labels (proposed)** | **0.9217 ± 0.0015** | **0.7878 ± 0.0052** | **0.7193 ± 0.0096** | **0.0348 ± 0.0093** | **0.0843 ± 0.0029** | **0.1514 ± 0.0060** |

Soft labels preserve discrimination and reduce the macro ECE by 22% relative to hard labels
(same sign in all five seeds; paired-bootstrap 95% CI of ΔECE: [−0.0111, −0.0072]).
All numbers, per-class results, paired tests and reliability-level analyses are in
[`results/stats.json`](results/stats.json).

## Repository layout

```text
ptb-xl-soft-labels/
├── README.md
├── requirements.txt
├── split_patient_level.csv      # ecg_id, partition (train/val/test), patient_id — 21,837 ECGs, 18,885 patients
├── train.py                     # trains one experiment from a JSON config (wraps src/pipeline.py)
├── evaluate.py                  # recomputes results/stats.json from the saved logits
├── configs/
│   ├── bce_hard.json            # Phase 2: Model B (SE+FFT), seeds 42–46, dropout 0.15
│   ├── bce_soft.json
│   ├── label_smoothing.json
│   ├── focal_loss.json
│   └── ablation/                # Phase 1: Models A/B × {base, se, se_fft, se_fft_attn}, seed 42
├── src/                         # config, data, models, engine (training/evaluation), pipeline
├── scripts/
│   ├── materialize_split.py     # builds train/ val/ test/ folders from split_patient_level.csv
│   └── check_split.py           # counts patients per partition and checks for leakage
└── results/
    ├── stats.json               # single source of every number in the paper
    ├── runs/<objective>/seed_<s>/{logits.npz, metricas.json}   # val/test logits and targets
    └── ablation/<A|B>_<variant>/metricas.json                  # Table III (validation)
```

## Installation

```bash
pip install -r requirements.txt
```

`torchaudio` is required: it implements the 0.5–120 Hz band-pass filter. Without it the filter is
disabled (a warning is printed) and results will not match the paper.

## Reproducing the paper

**1. Only the numbers (no GPU, no data download).** The saved logits contain everything needed:

```bash
python evaluate.py                     # writes results/stats.json (≈25 min with 2 processes)
python evaluate.py --jobs 8 --out /tmp/stats.json
```

This recomputes every table: metrics per seed, mean ± 95% CI, post-hoc temperature scaling for
all objectives, per-class calibration/discrimination, reliability-level MAE and entropy, and the
paired tests (paired t-test, exact Wilcoxon and paired bootstrap with 2,000 resamples, RNG seed 2026).

**2. Retraining.**

1. Download PTB-XL v1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2. Convert each record to `<ecg_id:05d>.pt` containing the tuple `(x, c)`:
   `x` = float tensor `(12, 5000)` (500 Hz signal) and `c` = float tensor `(5,)` with the
   cardiologist reliability (0–100) in the order `NORM, MI, STTC, CD, HYP`.
   <!-- TODO: add the preprocessing script used to build the tensors (scripts/build_tensors.py). -->
3. Materialize the split and train:

```bash
python scripts/materialize_split.py /path/to/tensors data/dataset_split
python train.py --config configs/bce_soft.json --data data/dataset_split           # 5 seeds
python train.py --config configs/ablation/B_se_fft.json --data data/dataset_split  # Phase 1
python train.py infer --config configs/bce_soft.json --data data/dataset_split --seed 42
```

Outputs are written to `runs/<config>/<arch>_<loss>/seed_<s>/` in the same format as `results/runs/`,
so `python evaluate.py --results <folder>` works on new runs as well.

Training used `cudnn.benchmark=True` (non-deterministic kernels): retraining reproduces the results
closely but not bit-for-bit. Exact reproduction of the reported numbers is guaranteed from the
saved logits (step 1).

## Experiment map

| Paper | Config(s) | Result files |
|---|---|---|
| Table I (split) | — | `split_patient_level.csv`, `results/runs/shared_bce_soft/particoes.json` |
| Table II (training setup) | any file in `configs/` | — |
| Table III (architecture / ablation) | `configs/ablation/*.json` | `results/ablation/*/metricas.json` → `stats.json["fase1"]` |
| Table IV (objectives, ± TS) | `configs/{bce_hard,bce_soft,label_smoothing,focal_loss}.json` | `stats.json["fase2"][<obj>]["macro"]` |
| Table V (paired comparisons) | same | `stats.json["testes"]` |
| Tables VI–VII, Fig. 2 (per class) | same | `stats.json["fase2"][<obj>]["classe"]` |
| Table VIII, Fig. 3 (reliability levels) | same | `stats.json["fase2"][<obj>]["faixa"]` |

## Patient-level split

| Subset | ECGs | Patients |
|---|---|---|
| Train | 15,235 | 13,219 |
| Validation | 3,322 | 2,833 |
| Test | 3,280 | 2,833 |
| **Total** | **21,837** | **18,885** |

No patient appears in more than one subset (`python scripts/check_split.py ptbxl_database.csv data/dataset_split`).

## Citation



## License

```text
MIT License

Copyright (c) 2026 Gabriel Andrade de Santana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
