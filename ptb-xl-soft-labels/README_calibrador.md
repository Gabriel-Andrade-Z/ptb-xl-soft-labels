# Calibrador Daheng MER2-041-528U3C

Controle, exibição e auto-calibração da câmera industrial Daheng
MERCURY2 MER2-041-528U3C (Sony IMX287) via Harvester SDK, focada em
maximizar nitidez para leitura de códigos de barras.

## Instalação

```bash
# 1) SDK Daheng (Galaxy) — instala driver e .cti
#    Baixe do site oficial: https://www.get-cameras.com/customerdownloads
#    O arquivo GxU3VTL.cti fica tipicamente em:
#      /opt/Daheng/Galaxy_camera/lib/GxU3VTL.cti           (Linux)
#      C:\Program Files\Daheng Imaging\GalaxySDK\...\GxU3VTL.cti (Windows)

# 2) Dependências Python
pip install harvesters numpy opencv-python

# 3) Permissões udev (Linux, USB3)
#    O instalador da Daheng costuma criar as regras. Confirme com:
lsusb | grep -i daheng
```

## Execução

```bash
# Usando o padrão do sistema
python3 daheng_barcode_calibrator.py

# Passando CTI explicitamente
python3 daheng_barcode_calibrator.py --cti /caminho/para/GxU3VTL.cti

# Via variável de ambiente
export DAHENG_CTI=/caminho/para/GxU3VTL.cti
python3 daheng_barcode_calibrator.py

# Com valores iniciais
python3 daheng_barcode_calibrator.py --exp 8000 --gain 6.0
```

## Atalhos

| Tecla | Ação |
|-------|------|
| **W** | Exposição +100 µs |
| **S** | Exposição −100 µs |
| **D** | Ganho +0.5 dB |
| **A** | Ganho −0.5 dB |
| **C** | Auto-calibração (Coarse-to-Fine) |
| **Q** / **Esc** | Sair |

Todos os valores são automaticamente **clampados** nas faixas oficiais
do datasheet: exposição 20 µs – 1 s, ganho 0 – 24 dB.

## Arquitetura (7 módulos)

| # | Classe / função | Responsabilidade |
|---|---|---|
| 1 | `CameraController` | Harvester + node_map; clamping; fetch com buffer.copy() + auto-queue via `with` |
| 2 | `ImagePipeline` | Fluxo duplo: cinza direto (métrica) / BGR VNG + croma CIELAB (anti-Moiré) |
| 3 | `SharpnessMetric` | Variância do Laplaciano (sem Gaussian) na ROI central 50% |
| 4 | `AutoCalibrator` | Coarse-to-Fine com descarte de 2 frames após cada set |
| 5 | `FPSLimiter` | Sleep dinâmico para teto de 30 FPS + EWMA para exibição |
| 6 | `draw_hud` | Retângulo da ROI + painel exp/gain/sharp%/FPS |
| 7 | `main` | Loop principal single-thread, teclas e encerramento seguro |

## Travas de estabilidade implementadas

- `ia.num_buffers = 2` — impede que a fila cresça e estoure a RAM.
- `.copy()` antes do buffer sair do `with` — nunca mantemos referência à
  memória pinned do driver.
- `fetch(timeout=1.0)` — nunca bloqueia indefinidamente; timeout devolve
  `None` e o loop continua responsivo.
- `time.sleep()` dinâmico para respeitar teto de 30 FPS — CPU baixa.
- Nome de janela ASCII puro (`"Calibrador Daheng"`) + `namedWindow`
  criado UMA vez fora do loop — evita bug de janelas duplicadas no
  OpenCV/Linux.
- Encerramento em `finally`: `cam.stop() → cam.close() → destroyAllWindows()`
  — recursos sempre liberados, inclusive em Ctrl+C.

## Bit depth (Bayer 8/10/12 bit)

O `PixelFormat` do sensor é detectado automaticamente:

- `BayerRG8` → uint8 direto.
- `BayerRG10` → uint16 (10 LSBs) → `>> 2` → uint8.
- `BayerRG12` → uint16 (12 LSBs) → `>> 4` → uint8.

## Testes

`test_calibrator.py` valida (sem hardware):

- Constantes contra o datasheet.
- Pipeline duplo (cinza / BGR VNG + CIELAB).
- Métrica de nitidez: ratio 1255x nítido/borrado + isolamento perfeito
  da ROI.
- Clamping de exposição e ganho.
- Auto-calibrador achando pico sintético conhecido (exp exata + gain
  a 2 dB), com **167 fetch_calls** confirmando descartes de 2 frames.
- FPSLimiter: 15 iterações em 503 ms → 29.8 FPS.
- Bit shift 12→8 e 10→8 preserva bits significativos corretos.
