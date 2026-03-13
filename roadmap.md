# Roadmap de Implementação — Polymarket Edge Explorer

## Contexto

Sistema quantitativo para explorar o **Favorite-Longshot Bias** na Polymarket.
Baseado na pesquisa de J. Becker sobre microestrutura de mercados de previsão:
- Preços > 0.80 são **subestimados** (probabilidade real maior que o preço)
- Preços < 0.20 são **superestimados** (probabilidade real menor que o preço)

**Stack:** Python 3.11+, PostgreSQL + TimescaleDB, Redis, Docker

---

## Status das Etapas

- [x] Infraestrutura base (Docker, PostgreSQL, TimescaleDB, Redis)
- [x] Schema inicial (`events`, `markets`, `market_outcomes`, `price_history`)
- [x] `gamma_api.py` — cliente REST para metadados de mercados ativos
- [x] `clob_api.py` — busca histórico de preço MID via CLOB
- [x] `markets.py` — scraper de eventos **ativos**
- [x] `price_worker.py` — consumidor Redis, insere histórico de preços
- [x] **Etapa 1** — Adaptação do Schema
- [x] **Etapa 2** — `historical_worker.py` (coleta de mercados resolvidos)
- [x] Etapa 3 — Captura de Bid-Ask Spread
- [x] Etapa 4 — Feature Pipeline
- [x] Etapa 5 — Modelagem ML (Logistic Regression + XGBoost) — `src/model_pipeline.py`
- [ ] Etapa 6 — Backtester Quantitativo
- [ ] Etapa 7 — Orquestração Docker (opcional)

---

## Etapa 1 — Adaptação do Schema

### Objetivo
Adicionar ao schema os campos necessários para armazenar resolução de mercados
(vencedor, timestamp, status) e dados de bid/ask, além de criar uma
Continuous Aggregate no TimescaleDB para pré-calcular volatilidade.

### Descobertas da API

A Gamma API retorna os seguintes campos em mercados fechados que são relevantes:

```json
{
  "outcomePrices": ["1", "0"],
  "outcomes":      ["Acend", "Bebop"],
  "clobTokenIds":  ["TOKEN_A", "TOKEN_B"],
  "closedTime":    "2026-01-23 20:56:36+00",
  "umaResolutionStatus": "resolved",
  "bestBid":       0.999,
  "bestAsk":       1.0,
  "spread":        0.001
}
```

- **Vencedor**: `outcomePrices[i] == "1"` → `winner_token_id = clobTokenIds[i]`
- **Bid/Ask histórico**: a CLOB API só retorna preço MID historicamente. Bid e Ask
  **só estarão disponíveis para dados novos** capturados a partir de agora via
  `bestBid`/`bestAsk` da Gamma API.

### Mudanças no Schema

**Arquivo a criar:** `schema/03_resolution.sql`

#### 1a. Campos de resolução na tabela `markets`

```sql
ALTER TABLE markets
  ADD COLUMN IF NOT EXISTS resolved_at           TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS winner_token_id       TEXT,
  ADD COLUMN IF NOT EXISTS uma_resolution_status TEXT;
```

| Coluna | Tipo | Fonte na API | Uso |
|---|---|---|---|
| `resolved_at` | `TIMESTAMPTZ` | `closedTime` | Ordenação temporal, feature `time_to_expiry` |
| `winner_token_id` | `TEXT` | derivado de `outcomePrices` + `clobTokenIds` | Target `y` do ML |
| `uma_resolution_status` | `TEXT` | `umaResolutionStatus` | Filtrar mercados ambíguos |

> **Por que `winner_token_id` sem FK?**
> A ordem de inserção é: (1) INSERT markets → (2) INSERT market_outcomes →
> (3) UPDATE markets SET winner_token_id. Uma FK quebraria no passo 1.
> A integridade é garantida pela lógica da aplicação.

#### 1b. Bid e Ask na tabela `price_history`

```sql
ALTER TABLE price_history
  ADD COLUMN IF NOT EXISTS bid NUMERIC,
  ADD COLUMN IF NOT EXISTS ask NUMERIC;
```

Colunas **nullable** — compatíveis com todos os dados históricos existentes
(que só têm preço MID). O `price_worker.py` não precisa de nenhuma alteração.

#### 1c. Continuous Aggregate para volatilidade diária

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS price_volatility_1d
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', time)  AS bucket,
  token_id,
  stddev(price)               AS price_std_1d,
  avg(price)                  AS price_avg_1d,
  count(*)                    AS sample_count
FROM price_history
GROUP BY bucket, token_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('price_volatility_1d',
  start_offset      => INTERVAL '7 days',
  end_offset        => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour'
);
```

O TimescaleDB atualiza esta view incrementalmente a cada hora. A feature pipeline
apenas lê a view — sem cálculos pesados no Pandas.

### Mudanças no Código

**Arquivo:** `src/db.py`

| Função | Mudança |
|---|---|
| `_flatten_market()` | Adicionar `resolved_at`, `winner_token_id`, `uma_resolution_status` |
| `insert_price_history()` | Aceitar `bid` e `ask` como parâmetros opcionais (default `None`) |
| `update_market_resolution()` | **Nova função** — UPDATE isolado, chamado após upsert de outcomes |

### Entrega da Etapa 1

```
schema/03_resolution.sql   ← novo arquivo SQL de migração
src/db.py                  ← 3 funções modificadas/adicionadas
```

---

## Etapa 2 — `historical_worker.py`

### Objetivo
Coletar todos os mercados já **resolvidos** da Polymarket, extrair o vencedor
de cada um, persistir no banco e enfileirar os `token_ids` para o
`price_worker.py` popular o histórico de preços.

### Descobertas da API

Investigação realizada em 12/03/2026 contra a API pública da Polymarket:

| Métrica | Valor |
|---|---|
| Total de mercados fechados | ~546.132 |
| Método de paginação | Offset-based (`limit` + `offset`) |
| Metadados de paginação no response | Nenhum — array JSON puro |
| Sinal de fim de dados | Array vazio `[]` |
| Requests necessários (limit=100) | ~5.462 |
| Tempo estimado (0.5s por request) | ~45 minutos |
| Rate limit explícito | Não encontrado nos headers |
| Cache Cloudflare | 3 minutos (`Cache-Control: max-age=180`) |

**Endpoint escolhido: `/events?closed=true&active=false`**

Escolhemos o endpoint de **eventos** (não o de mercados flat) por três razões:
1. Mantém a relação `event_id → market_id` que o schema exige (FK)
2. Inclui `tags[]` — necessário para a feature `market_category` na Fase ML
3. Os mercados aninhados já contêm `outcomePrices` + `clobTokenIds`

Os dados de `outcomePrices` encontrados na amostra são sempre strings binárias:
```
outcomePrices: ["1", "0"]  ou  ["0", "1"]
```
Nenhum caso de N/A (`["0.5","0.5"]`) foi encontrado na amostra de 10 mercados,
mas o código tratará esse edge case (→ `winner_token_id = NULL`).

### Fluxo de Execução

```
historical_worker.py
│
├── 1. Ler checkpoint do Redis
│      GET "historical_worker:last_offset" → offset inicial
│
├── 2. Loop de paginação
│      GET /events?closed=true&active=false&limit=100&offset=N
│      │
│      ├── Array vazio? → FIM (log de conclusão)
│      │
│      └── Para cada evento:
│            a. upsert_events()           ← reutiliza db.py
│            b. upsert_tags()             ← reutiliza db.py
│            c. link_tags_to_events()     ← reutiliza db.py
│            │
│            └── Para cada market no evento:
│                  d. upsert_markets()           ← reutiliza db.py
│                  e. upsert_market_outcomes()   ← reutiliza db.py
│                  f. extract_winner_token_id()  ← NOVA função (local)
│                  g. update_market_resolution() ← NOVA função em db.py
│                  h. enfileirar token_ids → Redis "price_worker_queue"
│
├── 3. Salvar checkpoint
│      SET "historical_worker:last_offset" = N + 100
│
├── 4. sleep(0.5)
│
└── 5. Próxima iteração (N += 100)
```

### Lógica Central: `extract_winner_token_id()`

```python
def extract_winner_token_id(market: dict) -> Optional[str]:
    """
    Determina o token_id vencedor de um mercado resolvido.
    Retorna None se a resolução for ambígua (ex: N/A) ou dados inválidos.

    A Gamma API representa o vencedor como o outcome com outcomePrices[i] == "1".
    Todos os valores são strings ("0" ou "1"), nunca floats.
    """
    try:
        prices    = json.loads(market.get("outcomePrices", "[]"))
        token_ids = json.loads(market.get("clobTokenIds", "[]"))

        for i, price_str in enumerate(prices):
            if float(price_str) == 1.0 and i < len(token_ids):
                return token_ids[i]
    except (json.JSONDecodeError, ValueError, IndexError):
        return None

    return None  # Resolução N/A ou ambígua — excluir do ML
```

### Checkpointing via Redis

Com ~5.500 requests necessários, retomada após falha é obrigatória.

```
Início do worker:
  offset = redis.get("historical_worker:last_offset") or 0

Após cada página processada com sucesso:
  redis.set("historical_worker:last_offset", offset + 100)

Reset manual (reprocessar tudo):
  redis.delete("historical_worker:last_offset")
```

### Dois Modos de Operação

**`--mode full` (coleta inicial única)**
- Varre todos os ~546K mercados do início ao fim
- Roda uma única vez como job isolado
- Checkpoint permite retomada em caso de falha

**`--mode incremental` (execução periódica)**
- Busca apenas mercados fechados recentemente
- Para quando `closedTime` do evento sair da janela de tempo configurável
- Pode ser agendado via cron ou loop com sleep no Docker Compose

### Arquivos a Criar/Modificar

| Arquivo | Tipo | O que muda |
|---|---|---|
| `src/historical_worker.py` | **Novo** | Lógica completa descrita acima |
| `src/db.py` | **Modificar** | Adicionar `update_market_resolution()` |
| `schema/03_resolution.sql` | **Prerequisito** | Deve ser executado antes (Etapa 1) |

### O que é Reutilizado vs Novo

```
historical_worker.py
├── REUTILIZA  PolymarketGammaClient.get_events()   → gamma_api.py
├── REUTILIZA  upsert_events()                       → db.py
├── REUTILIZA  upsert_markets()                      → db.py
├── REUTILIZA  upsert_market_outcomes()              → db.py
├── REUTILIZA  upsert_tags()                         → db.py
├── REUTILIZA  link_tags_to_events()                 → db.py
├── REUTILIZA  get_redis_connection()                → redis_client.py
├── REUTILIZA  get_db_connection()                   → db.py
├── NOVO       extract_winner_token_id(market)       → local em historical_worker.py
├── NOVO       update_market_resolution(conn, ...)   → adicionar em db.py
└── NOVO       lógica de checkpoint via Redis        → local em historical_worker.py
```

### Entrega da Etapa 2

```
src/historical_worker.py   ← novo arquivo principal
src/db.py                  ← adicionar update_market_resolution()
```

---

## Etapa 3 — Captura de Bid-Ask Spread

### Objetivo
Enriquecer o dataset com dados de bid e ask para uso no backtester
(custo real de entrada) e como feature de liquidez no modelo ML
(`spread_width = ask - bid`).

### Descobertas da API

Investigação realizada em 12/03/2026 contra a CLOB API pública:

**Endpoints disponíveis para bid/ask (tempo real):**

| Endpoint CLOB | Response | Uso |
|---|---|---|
| `/book?token_id=X` | Orderbook completo com `bids[]` e `asks[]` | Snapshot rico, 1 req/token |
| `/spread?token_id=X` | `{"spread": "0.018"}` | Spread instantâneo, leve |
| `/price?token_id=X&side=buy` | `{"price": "0.509"}` → **Ask** | Preço de compra (entrada) |
| `/price?token_id=X&side=sell` | `{"price": "0.491"}` → **Bid** | Preço de venda (saída) |
| `/prices-history` | Apenas `{t, p}` (MID) | **Sem bid/ask histórico** |

**Conclusão crítica:** Bid/ask histórico **não existe em nenhuma API pública da
Polymarket**. A `prices-history` retorna exclusivamente o preço MID. Só é
possível capturar bid/ask em tempo real, a partir de agora.

**Dados do py-clob-client confirmados:**
- `get_order_book(token_id)` → `OrderBookSummary` com `bids[]`, `asks[]`, `last_trade_price`
- `get_price(token_id, side)` → preço unitário por lado (leve, sem autenticação)
- `get_spread(token_id)` → spread direto
- Versões batch: `get_order_books()`, `get_prices()` — reduzem número de requests

**Todos os valores retornados são strings** (`"0.491"`, não `0.491`) — requerem
conversão explícita para `float`.

---

### Estratégia por Tipo de Dado

O problema fundamental é que o dataset histórico e o dataset futuro têm
disponibilidades diferentes:

```
Dados Históricos (já coletados via historical_worker.py)
└── bid/ask: NULL — não existe na API
    └── Fallback no feature pipeline: usar spread do raw_data JSONB do mercado
        SQL: COALESCE(ph.ask - ph.bid, (m.raw_data->>'spread')::numeric)

Dados Novos (a partir de agora, mercados ativos)
└── bid/ask: capturado em tempo real pelo orderbook_snapshot_worker.py
    └── Inserido em price_history com bid e ask preenchidos
```

---

### Componente Principal: `orderbook_snapshot_worker.py`

Novo serviço independente que captura snapshots periódicos de bid/ask
para todos os tokens ativos. Separa claramente as responsabilidades:

- `price_worker.py` → histórico MID (via `prices-history`)
- `orderbook_snapshot_worker.py` → snapshots live de bid/ask

**Fluxo de Execução:**

```
orderbook_snapshot_worker.py
│
├── 1. Buscar todos os token_ids ativos no banco
│      SELECT mo.token_id
│      FROM market_outcomes mo
│      JOIN markets m ON mo.market_id = m.id
│      WHERE m.active = true AND m.closed = false
│
├── 2. Para cada batch de token_ids (ex: 50 por vez)
│      │
│      ├── Chamar CLOB: /price?token_id=X&side=buy  → ASK
│      ├── Chamar CLOB: /price?token_id=X&side=sell → BID
│      └── Inserir em price_history:
│              time     = NOW()
│              token_id = X
│              price    = (ask + bid) / 2   # MID calculado
│              bid      = valor do /price?side=sell
│              ask      = valor do /price?side=buy
│              side     = 'SNAPSHOT'
│              amount   = NULL
│
├── 3. sleep(SNAPSHOT_INTERVAL_SECONDS)  # default: 900 (15 min)
│
└── 4. Repetir indefinidamente
```

**Por que `/price?side=X` em vez de `/book`?**
- `/price` = 2 requests por token (buy + sell) mas payload mínimo
- `/book` = 1 request por token mas retorna ~32 níveis de orderbook (pesado)
- Para o nosso objetivo (best bid / best ask), `/price` é suficiente e mais eficiente

---

### Adaptação do `markets.py` (mínima)

O `markets.py` já obtém `bestBid` e `bestAsk` da Gamma API a cada execução
(esses campos estão em `raw_data` JSONB). Nenhuma mudança estrutural necessária
— o `orderbook_snapshot_worker.py` vai capturar com mais granularidade
via CLOB.

**Opcional:** ao enfileirar market_ids no Redis, também publicar o `bestBid`/
`bestAsk` do momento como contexto para o snapshot worker. Avaliamos se há
ganho real antes de implementar.

---

### Estimativa de Spread para Dados Históricos

Para dados históricos sem bid/ask, a feature pipeline (Etapa 4) usará
fallback em cascata:

```sql
-- Prioridade 1: bid/ask real capturado pelo snapshot worker
ask - bid

-- Prioridade 2: spread do mercado (campo da Gamma API, salvo em raw_data)
(m.raw_data->>'spread')::numeric

-- Prioridade 3: mediana de spread de mercados similares (por tag/volume)
-- calculado como Continuous Aggregate ou view separada
```

Para o modelo ML, o spread histórico estimado tem baixa precisão mas captura
a ordem de magnitude correta. Mercados ilíquidos têm spreads maiores (>5%)
e líquidos têm spreads menores (<1%). Mesmo a estimativa grosseira é útil
como feature de liquidez.

---

### Constante de Fees para o Backtester

Descoberta da investigação: o endpoint CLOB `/markets/<condition_id>` expõe:

```json
"maker_base_fee": 0.002,
"taker_base_fee": 0.002
```

O taker fee (0.2%) é o custo relevante para o backtester, pois entradas via
market order sempre pagam taker fee. Esse valor é confirmado pela documentação
oficial. Portanto:

```
Custo real de compra = Ask_price × (1 + 0.002)
Payoff se ganhar     = 1.0 (resolução em $1 por share)
Payoff se perder     = 0.0
```

Este valor deve ser configurável como constante em `backtester.py`.

---

### Arquivos a Criar/Modificar

| Arquivo | Tipo | O que muda |
|---|---|---|
| `src/orderbook_snapshot_worker.py` | **Novo** | Worker de snapshots bid/ask |
| `src/clob_api.py` | **Modificar** | Adicionar `get_bid_ask(token_id)` |
| `src/db.py` | **Sem mudança** | `insert_price_history` já aceita bid/ask (Etapa 1) |
| `schema/03_resolution.sql` | **Prerequisito** | Colunas bid/ask já definidas (Etapa 1) |

### O que é Reutilizado vs Novo

```
orderbook_snapshot_worker.py
├── REUTILIZA  get_db_connection()           → db.py
├── REUTILIZA  insert_price_history()        → db.py (já aceita bid/ask)
├── REUTILIZA  get_redis_connection()        → redis_client.py
├── NOVO       get_active_token_ids(conn)    → nova query em db.py
├── NOVO       get_bid_ask(token_id)         → nova função em clob_api.py
└── NOVO       loop de snapshot periódico    → local em orderbook_snapshot_worker.py
```

### Entrega da Etapa 3

```
src/orderbook_snapshot_worker.py   ← novo worker de snapshots
src/clob_api.py                    ← adicionar get_bid_ask()
src/db.py                          ← adicionar get_active_token_ids()
```

---

## Etapa 4 — Feature Pipeline

### Objetivo
Transformar os dados brutos do TimescaleDB em um dataset tabular limpo
`(X, y)` pronto para treinar os modelos ML, exportado como `data/ml_dataset.parquet`.

### Descobertas da Investigação (12/03/2026)

**Densidade de dados confirmada:**

| Tipo de Mercado | Volume | Pontos de Preço | Intervalo Médio |
|---|---|---|---|
| Alta liquidez (LoL, BTC) | >$1M | ~13-17 por 2-3h | ~10 minutos |
| Zero volume | $0 | **0 pontos** | N/A |

Mercados de 2 dias: ~288 pontos por token (~6/hora × 48h).
Mercados zero-volume retornam `{"history": []}` — inutilizáveis para ML.

**`closedTime` ≠ `endDate` — diferença crítica:**

| Situação | Delta Observado |
|---|---|
| Resolução normal | +2h a +5h após `endDate` |
| Resolução antecipada | Até -7 dias antes de `endDate` |

**Conclusão:** `time_to_expiry` deve usar `resolved_at` (= `closedTime`) como
referência, não `end_date`. Para inferência em mercados ativos (não resolvidos),
usar `end_date` como proxy aceitável.

**Categorias disponíveis:**
O campo `category` no response da Gamma API já agrupa mercados em:
`"Sports"`, `"Crypto"`, `"US-current-affairs"`, `"Entertainment"` etc.
É mais confiável que parsear os 50 tags individuais (que são muito granulares
e concentrados em eleições de 2024).

---

### Unidade de Observação

Cada linha do dataset = **(token_id, snapshot_time)**.

Para mercados binários (YES/NO): 2 linhas por snapshot (uma por token).

**Estratégia de amostragem temporal:**
Usar todas as ticks de `price_history` é excessivo (até milhões de linhas no
agregado). A abordagem é **um snapshot diário por token** — mantém a dinâmica
temporal sem explodir o dataset:

```
Mercado de 30 dias → 30 snapshots × 2 tokens = 60 linhas
Mercado de 2 dias  → 2 snapshots × 2 tokens  = 4 linhas
```

Estimativa final do dataset: ~100K mercados resolvidos com histórico
× ~5 dias (mediana) × 2 tokens = **~1M linhas**, tamanho razoável.

---

### Features Definidas

| Feature | Tipo | Cálculo | Fonte |
|---|---|---|---|
| `implied_prob` | float | `price` MID diário médio | `price_history` |
| `time_to_expiry_days` | float | `(resolved_at - snapshot_date) / 86400` | `markets.resolved_at` |
| `price_std_1d` | float | stddev do preço no dia | `price_volatility_1d` (CA) |
| `price_std_7d` | float | média dos últimos 7 `price_std_1d` | `price_volatility_1d` (CA) |
| `spread_width` | float | `ask - bid` ou fallback do `raw_data` | `price_history` + `markets.raw_data` |
| `volume_total` | float | volume total do mercado | `markets.raw_data->>'volumeNum'` |
| `volume_1wk` | float | volume na última semana | `markets.raw_data->>'volume1wk'` |
| `is_sports` | bool | `category = 'Sports'` | `markets.raw_data->>'category'` |
| `is_crypto` | bool | `category = 'Crypto'` | `markets.raw_data->>'category'` |
| `is_politics` | bool | `category` contém "affairs" ou "Election" | `markets.raw_data->>'category'` |
| `outcome_index` | int | 0 (YES) ou 1 (NO) | `market_outcomes.outcome_index` |
| `y` | int | 1 se venceu, 0 se perdeu | `markets.winner_token_id` |

---

### Continuous Aggregate Adicional

`price_volatility_1d` (criada na Etapa 1) já fornece `price_std_1d`.
Para `price_std_7d`, nenhum novo CA é necessário: calculado no Python
com rolling window sobre os valores diários carregados do CA:

```python
df["price_std_7d"] = (
    df.groupby("token_id")["price_std_1d"]
      .transform(lambda s: s.rolling(7, min_periods=1).mean())
)
```

---

### Filtros de Qualidade de Dados

Aplicados via SQL antes de qualquer processamento Python:

```sql
-- 1. Apenas mercados realmente resolvidos com vencedor definido
WHERE m.uma_resolution_status = 'resolved'
  AND m.winner_token_id IS NOT NULL

-- 2. Excluir mercados sem histórico de preços
AND EXISTS (
    SELECT 1 FROM price_history ph
    WHERE ph.token_id = mo.token_id LIMIT 1
)

-- 3. Apenas snapshots ANTES da resolução (sem look-ahead)
AND snapshot_date < m.resolved_at::date

-- 4. time_to_expiry mínimo de 1 hora (evita snapshots no momento de resolução)
AND EXTRACT(EPOCH FROM (m.resolved_at - snapshot_ts)) > 3600
```

---

### Query SQL Principal

O banco faz o trabalho pesado; o Python apenas pós-processa:

```sql
WITH daily_snapshots AS (
  -- Um snapshot por (token, dia): usa a mediana diária do preço
  SELECT
    token_id,
    time_bucket('1 day', time)          AS snapshot_date,
    percentile_cont(0.5)
      WITHIN GROUP (ORDER BY price)     AS implied_prob,
    MAX(COALESCE(ask - bid, NULL))      AS spread_width_direct
  FROM price_history
  WHERE side IN ('MID', 'SNAPSHOT')
  GROUP BY token_id, snapshot_date
),

spread_fallback AS (
  -- Fallback de spread: usa o campo 'spread' do mercado salvo em raw_data
  SELECT
    mo.token_id,
    (m.raw_data->>'spread')::numeric    AS market_spread
  FROM market_outcomes mo
  JOIN markets m ON mo.market_id = m.id
),

volatility_7d AS (
  -- Média dos últimos 7 dias de desvio padrão
  SELECT
    token_id,
    bucket,
    price_std_1d,
    AVG(price_std_1d) OVER (
      PARTITION BY token_id
      ORDER BY bucket
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                   AS price_std_7d
  FROM price_volatility_1d
)

SELECT
  ds.token_id,
  ds.snapshot_date,
  ds.implied_prob,
  COALESCE(ds.spread_width_direct,
           sf.market_spread)            AS spread_width,
  EXTRACT(EPOCH FROM
    (m.resolved_at - ds.snapshot_date::timestamptz)
  ) / 86400.0                           AS time_to_expiry_days,
  v.price_std_1d,
  v.price_std_7d,
  (m.raw_data->>'volumeNum')::numeric   AS volume_total,
  (m.raw_data->>'volume1wk')::numeric   AS volume_1wk,
  (m.raw_data->>'category')            AS category,
  mo.outcome_index,
  CASE WHEN mo.token_id = m.winner_token_id
       THEN 1 ELSE 0 END               AS y

FROM daily_snapshots ds
JOIN market_outcomes mo  ON mo.token_id    = ds.token_id
JOIN markets m           ON m.id           = mo.market_id
LEFT JOIN spread_fallback sf ON sf.token_id = ds.token_id
LEFT JOIN volatility_7d v    ON v.token_id  = ds.token_id
                             AND v.bucket   = ds.snapshot_date

WHERE m.uma_resolution_status = 'resolved'
  AND m.winner_token_id IS NOT NULL
  AND ds.snapshot_date < m.resolved_at::date
  AND EXTRACT(EPOCH FROM
        (m.resolved_at - ds.snapshot_date::timestamptz)) > 3600

ORDER BY ds.snapshot_date, ds.token_id;
```

---

### Pós-processamento em Python

Após carregar o resultado da query:

```
feature_pipeline.py
│
├── 1. Executar query SQL → carregar em pandas DataFrame
│
├── 2. Computar price_std_7d via rolling window (por token_id)
│
├── 3. Encoding de categorias
│      category → one-hot: is_sports, is_crypto, is_politics, is_other
│      (baseado no campo 'category' do raw_data, não nos 50 tags individuais)
│
├── 4. Imputação de valores ausentes
│      spread_width → 0.002 (mediana de mercado líquido) quando NULL após fallback
│      price_std_1d → 0.0 quando mercado sem histórico suficiente
│      volume_total → 0.0 quando ausente
│
├── 5. Validação anti-leakage
│      Assert: snapshot_date < resolved_at para TODAS as linhas
│      Assert: implied_prob ∈ [0.001, 0.999]
│      Assert: time_to_expiry_days > 0 para TODAS as linhas
│      Log: distribuição de y (% vencedores — deve ser ~50% para mercados binários)
│
├── 6. Exportar para Parquet
│      data/ml_dataset.parquet  (compressão snappy)
│      data/ml_dataset_meta.json  (estatísticas: linhas, colunas, data range)
│
└── 7. Imprimir relatório de qualidade
       Total de linhas, mercados únicos, tokens únicos
       % de linhas com spread real vs estimado
       Distribuição de y por categoria
```

---

### Decisão Chave: Por que `category` em vez dos 50 tags individuais?

Os 50 tags investigados são excessivamente granulares e enviesados:
- 15 dos top 20 tags referem-se a candidatos/eventos da eleição americana de 2024
- Tags como "Trump", "Kamala Harris" não generalizam para outros mercados
- Muitos mercados esportivos têm 0 tags

O campo `category` (da Gamma API, salvo em `raw_data` JSONB) agrega tudo
em ~5 categorias estáveis. Para o ML, isso é mais robusto e generalizável.

---

### Arquivos a Criar

| Arquivo | Tipo | Conteúdo |
|---|---|---|
| `src/feature_pipeline.py` | **Novo** | Script completo de extração e export |
| `data/` | **Novo diretório** | Output do parquet (não versionado no git) |

### Sem mudanças em arquivos existentes

A feature pipeline é puramente de **leitura** — não modifica nenhuma tabela.
Apenas consulta e exporta.

### Entrega da Etapa 4

```
src/feature_pipeline.py      ← script principal
data/ml_dataset.parquet      ← dataset gerado (gitignored)
data/ml_dataset_meta.json    ← metadados do dataset (linhas, período, etc.)
```

---

## Etapa 5 — Modelagem ML e Calibração de Probabilidades

### Objetivo
Treinar modelos que produzam probabilidades **bem calibradas** — isto é, quando
o modelo diz 85%, o evento deve ganhar ~85% das vezes. A calibração é o que
transforma previsões em edge real no backtester.

Responder à pergunta central: **o Favorite-Longshot Bias existe nos nossos
dados?** Se sim, um modelo calibrado consegue extrair valor onde o mercado
está sistematicamente errado.

---

### Propriedade do Dataset: Balanceamento Perfeito

Para cada mercado binário (YES/NO), há exatamente 1 vencedor e 1 perdedor.
Como cada mercado gera 2 linhas no dataset (uma por token), o target `y` é
**exatamente 50% positivo / 50% negativo por construção**.

Consequência direta: **não há necessidade de class weighting, SMOTE ou
qualquer técnica de rebalanceamento**. Isso simplifica o pipeline.

---

### Divisão Temporal — Out-of-Time Split

A divisão é **por mercado** (não por snapshot), ordenada por `resolved_at`.

```
Todos os mercados ordenados por resolved_at:
│
├── Treino:  80% mais antigos  (ex: Jan 2022 → Jun 2025)
│
└── Teste:   20% mais recentes (ex: Jul 2025 → Mar 2026)
```

**Por que dividir por mercado e não por snapshot_date?**
Se dividíssemos por data de snapshot, o mesmo mercado poderia aparecer em
treino (snapshot T-30d) e em teste (snapshot T-1d). Isso não é leakage estrito,
mas contamina a avaliação: o modelo veria o mesmo mercado em contextos diferentes.
Dividir por mercado garante separação total entre treino e teste.

**Implementação:**
```python
# Pegar os market_ids únicos ordenados por data de resolução
market_dates = df.groupby("market_id")["resolved_at"].max().sort_values()
cutoff_idx   = int(len(market_dates) * 0.80)
train_markets = set(market_dates.iloc[:cutoff_idx].index)
test_markets  = set(market_dates.iloc[cutoff_idx:].index)

df_train = df[df["market_id"].isin(train_markets)]
df_test  = df[df["market_id"].isin(test_markets)]
```

---

### Modelo 1 — Baseline: Regressão Logística (Platt Scaling)

**Objetivo específico:** detectar o FLB com apenas a feature `implied_prob`.

```
Input:  X = [implied_prob]     (apenas o preço de mercado)
Output: calibrated_probability  (probabilidade ajustada pelo modelo)
```

**Pipeline:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  CalibratedClassifierCV(
                   LogisticRegression(max_iter=1000),
                   method="sigmoid",   # Platt Scaling
                   cv=5
               ))
])
baseline.fit(X_train[["implied_prob"]], y_train)
```

**O que o Reliability Diagram vai revelar:**

Se o FLB existir nos dados:
```
Probabilidade real
1.0 │                          ╭──── Diagonal perfeita
    │                     ╭────╯ ← Favoritos: mercado subestima (linha ACIMA)
0.8 │               ╭─────╯
    │          ╭────╯
0.6 │     ╭────╯
    │╭────╯ ← Azarões: mercado superestima (linha ABAIXO)
0.2 │
    └──────────────────────────── Probabilidade predita (preço)
       0.2  0.4  0.6  0.8  1.0
```

---

### Modelo 2 — Avançado: XGBoost Classifier

**Input:** todas as features da Etapa 4.

```python
FEATURE_COLS = [
    "implied_prob",
    "time_to_expiry_days",
    "price_std_1d",
    "price_std_7d",
    "spread_width",
    "volume_total",
    "volume_1wk",
    "is_sports",
    "is_crypto",
    "is_politics",
    "outcome_index",
]
```

**Tuning com TimeSeriesSplit sobre os dados de treino:**
```python
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

tscv = TimeSeriesSplit(n_splits=5)  # split temporal dentro do treino

param_grid = {
    "n_estimators":     [200, 500],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
}
```

**Early stopping para evitar overfitting:**
```python
xgb = XGBClassifier(
    objective      = "binary:logistic",
    eval_metric    = "logloss",
    early_stopping_rounds = 30,
    random_state   = 42,
)
xgb.fit(
    X_train, y_train,
    eval_set    = [(X_val, y_val)],
    verbose     = False,
)
```

**Calibração pós-treino:** o XGBoost já produz probabilidades via sigmoid
internamente, mas aplicar `CalibratedClassifierCV(method='isotonic')` melhora
a calibração em dados de teste, especialmente nas caudas (região FLB):
```python
xgb_calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
xgb_calibrated.fit(X_val, y_val)
```

---

### Métricas de Avaliação

Calculadas no conjunto de **teste** (out-of-time):

| Métrica | O que mede | Fórmula |
|---|---|---|
| **Log Loss** | Qualidade de calibração geral | `-mean(y·log(p) + (1-y)·log(1-p))` |
| **Brier Score** | Erro quadrático médio das probabilidades | `mean((y - p)²)` |
| **Expected Value (EV)** | Edge médio por aposta simulada | `mean(p_model - p_ask)` onde `p_model > p_ask` |
| **EV @ FLB Zone** | Edge específico na zona de favoritos | Filtrando `implied_prob > 0.80` |

**Fórmula de EV por trade:**
```
EV_por_trade = p_model - p_ask

Se EV > 0: o modelo acredita que o mercado está precificando errado.
Payoff realizado = 1 - p_ask  (se ganhar) ou  -p_ask  (se perder)
```

**Baseline de comparação (mercado eficiente):**
Se o mercado fosse perfeito, `p_model ≈ p_ask` e `EV ≈ 0`.
Log Loss do mercado puro: calculado usando `implied_prob` diretamente como
predição (sem modelo), servindo de benchmark mínimo.

---

### Análise de Feature Importance (XGBoost)

Após treino, gerar dois plots:
1. **SHAP values** por feature (impacto médio absoluto no output)
2. **Feature importance** nativa do XGBoost (`gain` e `cover`)

Hipóteses a verificar:
- `implied_prob` deve ser a feature mais importante (base do FLB)
- `time_to_expiry_days` deve ser relevante (bias varia com o tempo)
- `spread_width` deve capturar liquidez (mercados ilíquidos = preços menos confiáveis)

---

### Fluxo Completo do `model_pipeline.py`

```
model_pipeline.py
│
├── 1. Carregar data/ml_dataset.parquet
│
├── 2. Out-of-Time Split por market_id (80/20 cronológico)
│
├── 3. Modelo Baseline (LR + Platt Scaling)
│      a. Treinar em X_train[["implied_prob"]]
│      b. Avaliar em X_test: Log Loss, Brier Score
│      c. Gerar Reliability Diagram → reports/reliability_diagram_lr.png
│      d. Calcular EV médio no test set
│
├── 4. Modelo XGBoost
│      a. TimeSeriesSplit CV para tuning de hiperparâmetros (dentro de train)
│      b. Treinar melhor modelo com early stopping
│      c. Calibrar com CalibratedClassifierCV (isotonic) no val set
│      d. Avaliar em X_test: Log Loss, Brier Score, EV, EV@FLB
│      e. Gerar Reliability Diagram → reports/reliability_diagram_xgb.png
│      f. Plot SHAP → reports/feature_importance.png
│
├── 5. Comparar modelos em tabela consolidada
│      Baseline LR │ XGBoost │ Mercado Puro (benchmark)
│      Log Loss    │  ...    │  ...
│      Brier Score │  ...    │  ...
│      EV médio    │  ...    │  ...
│
└── 6. Serializar melhor modelo
       models/lr_baseline.pkl         ← joblib
       models/xgb_model.pkl           ← joblib
       models/feature_list.json       ← lista de features na ordem correta
       models/train_cutoff_date.txt   ← data de corte (para auditoria)
```

---

### Proteções Contra Data Leakage

| Risco | Proteção implementada |
|---|---|
| Usar dados futuros no treino | Split por `resolved_at` do mercado, não por snapshot |
| Feature calculada com dados do futuro | Assert `snapshot_date < resolved_at` na Etapa 4 |
| Calibração com dados de teste | XGBoost calibrado no `val set` (subset do treino), não no teste |
| Tuning de hiperparâmetros vendo o teste | `TimeSeriesSplit` apenas dentro de `df_train` |
| Mesmo mercado em treino e teste | Split feito por `market_id`, não por row |

---

### Arquivos a Criar

| Arquivo | Tipo | Conteúdo |
|---|---|---|
| `src/model_pipeline.py` | **Novo** | Pipeline completo de treino e avaliação |
| `models/` | **Novo diretório** | Modelos serializados (gitignored) |
| `reports/` | **Novo diretório** | Gráficos de calibração e feature importance |

### Entrega da Etapa 5

```
src/model_pipeline.py                    ← script principal
models/lr_baseline.pkl                   ← modelo baseline serializado
models/xgb_model.pkl                     ← modelo XGBoost serializado
models/feature_list.json                 ← features em ordem correta
models/train_cutoff_date.txt             ← data de corte do treino
reports/reliability_diagram_lr.png       ← curva de calibração baseline
reports/reliability_diagram_xgb.png      ← curva de calibração XGBoost
reports/feature_importance.png           ← SHAP values
```
