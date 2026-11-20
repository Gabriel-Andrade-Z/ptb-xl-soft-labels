# barcode_reader.py — Integração Leitor Industrial ↔ AI-VISION

Script único (`barcode_reader.py`) que substitui a versão anterior de leitura
por câmera industrial. Passa a se comportar como o **leitor da esteira** do
sistema AI-VISION: cada código lido dispara o pipeline de inspeção no
Backend.

## Requisitos

```
python -m pip install requests opencv-python pyzbar numpy pygobject
apt install libaravis-0.8-0 libzbar0        # dependências nativas
```

O Aravis exige `PyGObject` compilado contra as bibliotecas Aravis do sistema
(o mesmo caminho do seu script original).

## Configuração (variáveis de ambiente)

Nenhuma edição de código é necessária para trocar de ambiente:

| Variável | Padrão | Uso |
|----------|--------|-----|
| `AIVISION_URL`         | `http://localhost:8000` | Base do backend |
| `AIVISION_USER`        | `op01` | Operador do turno |
| `AIVISION_PASSWORD`    | `op01` | Senha do operador |
| `AIVISION_DEDUP_SEC`   | `3.0`  | Janela em que o mesmo código é ignorado |
| `AIVISION_EXPOSURE_US` | `8000` | Exposição da câmera (µs) |
| `AIVISION_GAIN`        | `2.0`  | Ganho analógico |

Exemplo de execução em chão de fábrica:

```
AIVISION_URL="http://192.168.10.20:8000" \
AIVISION_USER="op03" AIVISION_PASSWORD="op03" \
AIVISION_DEDUP_SEC=5 \
python3 barcode_reader.py
```

## Controles

- **q** ou **Esc** → sair
- A janela abre como redimensionável (`cv2.WINDOW_NORMAL`).
- O botão **"r" para cadastrar produto** foi **removido** — o cadastro
  agora acontece no servidor (o `produtos.json` local não é mais usado).
  Toda peça vem da importação do arquivo do ERP (`Star.xls`).

## Impactos no lado do servidor

**Nenhum arquivo do backend foi alterado**. O leitor consome apenas
endpoints que já existiam:

| Rota | Método | Uso pelo leitor |
|------|--------|-----------------|
| `/login`                   | POST | autenticação inicial e renovação de token |
| `/pipeline/piece-scanned`  | POST | envia cada barcode lido |
| `/health`                  | GET  | sanity check no startup |

O `PipelineService` no backend (`backend/app/services/pipeline_service.py`)
já era o ponto de entrada esperado — ele orquestra os 4 modelos de IA
(barcode → dimensional → visual → decisão), atualiza a peça no banco,
promove auto-transição de estado do pedido (`queued → inspecting →
finished`), cria `InspectionHistory` e `PieceResult`. O Dashboard e a
tela de Histórico atualizam sozinhos via polling.

Para operar em produção, deixe o backend rodando (`./run.sh` no diretório
do projeto) e execute o `barcode_reader.py` na máquina onde a câmera
industrial está conectada. Se estiverem em máquinas diferentes, aponte
`AIVISION_URL` para o IP do servidor.

---

# Relatório da arquitetura do código

O script está organizado em **7 módulos numerados** dentro do mesmo
arquivo. Cada um tem uma responsabilidade única e é substituível.

## 1. `BackendClient` — cliente HTTP com autenticação

Encapsula toda a comunicação com o servidor. Responsabilidades:

- **Cachear** o token JWT após o primeiro login.
- **Renovar** automaticamente ao receber `401` (token expirado): faz
  novo login e re-envia a requisição uma vez.
- **Timeout curto** (5 s) para nunca ficar preso quando o servidor
  fica lento ou sai do ar.
- Traduz cada resposta HTTP em um `ScanOutcome` (dataclass imutável).
  Nunca lança exceção para o loop de vídeo — falhas viram
  `ok=False, message="..."`.

Códigos de status tratados: `200` → sucesso; `401` → relogin; `404` →
peça não cadastrada; `409` → pedido já finalizado; outros → mensagem
genérica.

## 2. `BarcodeDeduplicator` — janela temporal

Impede que o **mesmo código** seja enviado dezenas de vezes por segundo.
Cada peça, quando passa na frente da câmera, é visível por vários
frames — sem dedup, o backend receberia N chamadas idênticas.

Estratégia: para cada barcode, guarda o timestamp da última liberação.
`should_send()` só devolve `True` se passou mais que `window_sec` desde
a última vez. **A marcação é feita dentro do mesmo lock que a consulta**,
evitando corrida entre "verificar" e "marcar" quando várias threads
consultam.

`prune()` é chamado a cada 100 envios para liberar memória de códigos
antigos.

## 3. `ScanDispatcher` — thread worker + fila

Este é o ponto que resolve o problema de FPS. O loop de vídeo precisa
rodar em ~30 FPS (~33 ms por frame). Um POST HTTP demora tipicamente
10–200 ms — se o loop esperasse a resposta, o vídeo travaria.

Solução: uma **thread separada** consome uma `queue.Queue` limitada. O
loop de vídeo chama `submit(barcode)`, que:

1. Consulta o dedup — se não deve enviar, retorna imediatamente.
2. Enfileira o barcode — retorna sem esperar.
3. Se a fila está cheia (backend caiu / rede muito lenta), descarta com
   log em stderr.

A thread worker consome a fila em loop, chama `client.submit_barcode()`,
guarda o último `ScanOutcome` para o overlay ler. Métrica de sanidade:
no teste, **65 chamadas a `submit()` completaram em 0,07 ms**.

`stop()` envia uma sentinela pela fila para desbloquear o worker e faz
`join()` com timeout — encerramento limpo.

## 4. `FPSCounter` — média deslizante (EWMA)

Contador de FPS simples e estável. Cada `tick()` calcula o FPS
instantâneo e combina com uma exponentially weighted moving average
(α = 0.1). Sem picos de "60.0" seguidos de "5.0" quando o loop varia.

## 5. Processamento de imagem

Reaproveita a lógica original (CLAHE + sharpening) como **fallback**:
tenta decodificar direto; se falhar, aplica pré-processamento e tenta
de novo. Isso segura casos de iluminação ruim ou desfoque leve, mas
não paga o custo do pré-processamento em toda frame.

`buffer_to_gray_and_bgr()` isola a conversão de formatos Bayer
(RG8/RG10/RG12) e mono num único ponto — se o modelo da câmera mudar,
só esta função precisa evoluir.

## 6. Overlay da janela

Duas funções:

- `draw_detection_boxes()` — desenha **apenas o polígono verde**
  (BGR `0,255,0`, espessura 3) ao redor de cada código. **Sem texto,
  sem rótulo** — conforme pedido.
- `draw_hud()` — canto superior esquerdo, uma única linha compacta com
  fundo preto:

  ```
  FPS 27.3  |  fila 0  |  env 12 err 1  |  ult: OK 000002 approved
  ```

  Cor da linha muda com o resultado (verde ok / vermelho erro). Fica
  fora da área útil da imagem, não polui o campo de visão da caixa.

## 7. Câmera Aravis

`open_camera()` isola toda a configuração: região do sensor,
exposição, ganho, WB automático (com `try/except` para câmeras mono
que não suportam), formato de pixel, criação do stream com 5 buffers
pré-alocados. Retorna a câmera pronta para `pop_buffer()`.

## Fluxo completo — como as peças se encaixam

```
┌──────────────────────────────────────────────────────────────────┐
│ Thread principal (loop de vídeo)                                 │
│                                                                  │
│  camera.pop_buffer()                                             │
│      ↓                                                           │
│  buffer_to_gray_and_bgr()                                        │
│      ↓                                                           │
│  stream.push_buffer(buffer)  ← devolve rápido ao stream          │
│      ↓                                                           │
│  decode_stable()                                                 │
│      ↓                                                           │
│  draw_detection_boxes()  ← só polígono verde                     │
│      ↓                                                           │
│  para cada barcode: dispatcher.submit(bc)  ← não bloqueia        │
│      ↓                                                           │
│  fps_counter.tick(); draw_hud(); cv2.imshow()                    │
│      ↓                                                           │
│  waitKey(1)  ← q/esc? sai                                        │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │  (fila thread-safe)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Thread worker (ScanDispatcher._run)                              │
│                                                                  │
│  bc = fila.get(timeout=0.5)                                      │
│      ↓                                                           │
│  BackendClient.submit_barcode(bc)                                │
│      ↓  (POST /pipeline/piece-scanned, com relogin em 401)       │
│  ScanOutcome → guarda como last_outcome para o HUD ler           │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │  HTTP JSON
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Backend AI-VISION                                                │
│                                                                  │
│  PipelineService.process_piece_scan(barcode)                     │
│      Modelo A (barcode) → identifica peça no BD                  │
│      Modelo B (dimensional) → mede                               │
│      Modelo C (visual) → busca defeitos                          │
│      Decisão consolidada → PieceResult + InspectionHistory       │
│      Auto-transição de estado do pedido                          │
│      ↓                                                           │
│  Dashboard (polling 3s) reflete a mudança                        │
└──────────────────────────────────────────────────────────────────┘
```

## Decisões de design que se pagam depois

**Configuração por env var, não hardcode.** O script sobe idêntico em
dev, staging e chão de fábrica — só muda o `AIVISION_URL`.

**BackendClient nunca lança.** Toda falha vira um `ScanOutcome(ok=False,
message=...)`. O loop de vídeo continua rodando mesmo com o backend
fora do ar; o HUD mostra o erro; ao restaurar o backend, o próximo
código lido passa normalmente.

**Dedup separado do dispatcher.** Dá pra trocar a política de
deduplicação (por exemplo: hash + tolerância a leituras próximas no
mesmo lote) sem mexer no dispatcher.

**Um único ponto de conversão de imagem.** Se o modelo da câmera mudar
(sensor diferente, formato de pixel novo), só `buffer_to_gray_and_bgr()`
precisa evoluir.

**Nada bloqueia o loop de vídeo.** O overlay lê `dispatcher.pending`,
`dispatcher.sent_count` e `dispatcher.last_outcome` — todos são acessos
atômicos por conta do GIL, sem lock. Não há await, não há
`response.wait()`.

## Migração para SSE / WebSockets (quando houver)

Hoje o Dashboard descobre atualizações por polling a cada 3 s. Quando o
backend ativar SSE via `/events/stream` (já existe stub), o
`ScanDispatcher` pode ganhar um observer que empurra evento local
imediatamente — mas o script atual **já é compatível**, porque o
Dashboard vai reagir em tempo real de qualquer forma. Nenhuma mudança
é necessária aqui.

## Limitações conhecidas

- **`produtos.json` não é mais usado.** O cadastro passou 100% pro
  servidor. Se você depende de peças cadastradas localmente, migre-as
  por meio de um arquivo do ERP importado via `POST /orders/import`.
- **Tecla `r` foi removida.** Cadastro pela UI do Backend
  (Configuração → Importar pedidos).
- **Não há autorização de operador por câmera.** Todo `barcode_reader.py`
  em execução usa o mesmo `AIVISION_USER`. Para rastrear operadores
  por estação, subir uma instância por câmera com `AIVISION_USER`
  diferente.

## Como testar sem câmera

O arquivo `test_barcode_reader.py` (junto com esta entrega) exercita
`BackendClient`, `BarcodeDeduplicator` e `ScanDispatcher` contra o
backend real, sem depender de Aravis/OpenCV. Serve tanto como
regressão quanto como referência de uso das classes.
