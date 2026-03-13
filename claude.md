# Projeto: Polymarket Edge Explorer (Favorite-Longshot Bias)

## Contexto do Projeto e Objetivo
Este projeto visa construir um sistema quantitativo para explorar ineficiências de precificação na Polymarket (Prediction Markets). Baseado na pesquisa de J. Becker sobre a microestrutura de mercados de previsão, exploraremos o fenômeno **Favorite-Longshot Bias**:
- Preços acima de 0.80 ($80\%$) são estatisticamente **subestimados** (a probabilidade real é maior que o preço reflete).
- Preços abaixo de 0.20 ($20\%$) são estatisticamente **superestimados** (a probabilidade real é menor que o preço reflete).

O objetivo é coletar dados, extrair features, treinar modelos de calibração de probabilidade (Regressão Logística $\rightarrow$ XGBoost) e desenvolver um backtester realista que valide a existência de lucro (Edge/EV positivo), considerando spread de compra/venda, taxas e tempo de lockup do capital.

## Stack Tecnológica
- **Infraestrutura:** Docker, Docker Compose
- **Banco de Dados:** PostgreSQL com extensão TimescaleDB (para séries temporais)
- **Mensageria:** Redis
- **Linguagem Principal:** Python 3.11+
- **Machine Learning:** `scikit-learn` (Regressão Logística), `xgboost`, `pandas`, `numpy`
- **APIs:** Polymarket Gamma API (metadados), Polymarket CLOB API (orderbook/preços)
- **Base/Scraper:** `Timon33/polymarket_scraper` (já existente no repositório, mas que sofrerá adaptações).

---

## Instruções para a IA (Claude Code / Coder Agent)
Aja como um Engenheiro Quantitativo Sênior e Engenheiro de Dados. Execute as tarefas abaixo em ordem. Para cada passo concluído, atualize o status. Se encontrar bloqueios, sugira alternativas baseadas nas melhores práticas de quant trading. **Sempre use validação Out-of-Time (Time-Series Split) para treinos de ML para evitar data leakage.**

### Fase 1: Adaptação da Coleta de Dados (Scraping Histórico)
O scraper original (`Timon33/polymarket_scraper`) foca em mercados ativos. Precisamos do histórico de mercados resolvidos para obter nossa variável alvo ($y$).
- [ ] Analisar o código do worker de extração atual.
- [ ] Criar um novo script/worker (`historical_worker.py`) que consulte a API da Polymarket em busca de mercados já resolvidos (closed/resolved).
- [ ] Salvar os metadados desses mercados resolvidos, incluindo qual `token_id` foi o vencedor (para criarmos $y=1$ para o vencedor e $y=0$ para os perdedores).
- [ ] Coletar o histórico de preços (Time Series) desses mercados resolvidos via CLOB API e popular o TimescaleDB.
- [ ] **Crucial:** Adaptar o scraper de preços para capturar ou estimar o **Bid-Ask Spread**, não apenas o "Last Traded Price", pois usaremos o preço de "Ask" (venda) para calcular o custo real de entrada no backtest.

### Fase 2: Engenharia de Dados e Features
Criar um pipeline em Python/SQL que transforme os dados brutos do TimescaleDB em um dataset tabular para ML (`X, y`).
- [ ] **Target ($y$):** 1 se a opção venceu, 0 se perdeu.
- [ ] **Feature Base ($x_1$):** Preço (Probabilidade implícita da Polymarket).
- [ ] **Feature:** `time_to_expiry` (Tempo restante até a resolução do mercado, em dias/horas).
- [ ] **Feature:** `price_volatility` (Desvio padrão do preço nas últimas 24h/7d).
- [ ] **Feature:** `liquidity_proxy` ou `spread_width` (Diferença entre Bid e Ask).
- [ ] **Feature:** `market_category` (One-hot encoding das tags, ex: Política, Cripto).
- [ ] Desenvolver um script que exporta esse dataframe final limpo para um arquivo `.parquet` ou `.csv` para facilitar a modelagem.

### Fase 3: Modelagem e Calibração de Probabilidades
Criar o pipeline de treinamento (`model_pipeline.py`). Dividir os dados ordenando pelo tempo (Treino no passado, Teste no futuro) - não usar Random Split.
- [ ] **Modelo Baseline:** Regressão Logística (Platt Scaling). Treinar o modelo usando apenas a feature de preço original. Analisar a curva de calibração (Reliability Diagram) para ver se o "Favorite-Longshot Bias" aparece nos nossos dados.
- [ ] **Modelo Avançado:** XGBoost Classifier. Treinar usando todas as features criadas na Fase 2. Otimizar hiperparâmetros cuidando com o overfitting.
- [ ] **Métricas de Avaliação:** Log Loss, Brier Score e Expected Value (EV) gerado.

### Fase 4: Motor de Backtesting Quantitativo
Não basta o modelo acertar; a estratégia precisa dar dinheiro na vida real.
- [ ] Criar `backtester.py` que simule a passagem do tempo.
- [ ] **Regra de Entrada:** Se `Modelo_Probabilidade > (Preço_Ask + Threshold_Edge)`, executar aposta "YES". (Focar especialmente nos casos onde Preço_Ask > 0.80).
- [ ] **Fatores Restritivos a codificar no simulador:**
  - Descontar fees (taxas de transação/rede).
  - Usar o preço `Ask` (o mais caro) para comprar, não o preço médio ou de último negócio.
  - Calcular o tempo de Lockup: Se compro hoje e resolve daqui a 30 dias, o capital fica travado.
- [ ] **Métricas de Saída:** PnL Total (Lucro/Prejuízo), Win Rate, Drawdown Máximo, Sharpe Ratio e **ROI Anualizado**.

### Fase 5: Preparação para Execução (Opcional / Futuro)
- [ ] Criar um wrapper simplificado para conectar o modelo final ao sistema de Order Book da Polymarket via CLOB API usando chaves de API reais, escaneando o mercado em tempo real em busca de apostas com `EV > 0`.

---
## Regras de Código e Commits
- Use tipagem estática do Python (`typing`).
- Escreva docstrings padrão para funções matemáticas ou de engenharia de features complexas.
- Ao usar SQL para o TimescaleDB, utilize Continuous Aggregates sempre que precisar calcular médias ou volatilidades temporais pesadas, deixando o banco fazer o trabalho pesado ao invés do Pandas.