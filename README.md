# TC5 — Plataforma de Experimentação Adaptativa de Ofertas

Projeto desenvolvido para o **Datathon da POSTECH/FIAP — MLET (Machine
Learning Engineering)**. O desafio propõe construir, de ponta a ponta, uma
plataforma de experimentação adaptativa que decide qual oferta apresentar a
um cliente de uma instituição financeira digital, no lugar de regras fixas
ou testes A/B longos.

## Status atual

| Etapa | Descrição | Status |
|---|---|---|
| 0 | Organização do projeto | ✅ Concluída |
| 1 | Base Kaggle e EDA | ✅ Concluída |
| 2 | Preparação da base | ✅ Concluída |
| 3 | Baseline e estratégia algorítmica | ✅ Concluída |
| 4 | Avaliação e Golden Set | ✅ Concluída |
| 5 | Serviço ou interface demonstrável | ✅ Concluída |
| 6 | Arquitetura-alvo em nuvem | ✅ Concluída |
| 7 | Ciclo de vida MLOps (MLflow) | ⏳ Pendente |
| 8 | Apresentação final (Demo Day) | ⏳ Pendente |

## Problema de negócio

Uma instituição financeira digital precisa decidir, em diferentes canais,
qual oferta, mensagem ou próximo passo apresentar para cada cliente
elegível. Regras fixas e testes A/B longos desperdiçam tráfego, demoram a
reagir a mudanças de contexto e dificultam a personalização responsável.
Uma abordagem adaptativa — como **multi-armed bandit** ("bandido de
múltiplos braços", em referência às antigas caça-níqueis de uma alavanca só)
— permite identificar comportamentos distintos, equilibrar exploração e
explotação, e aprender com respostas observadas sem congelar a decisão em
regras estáticas.

## Abordagem

| Componente | Escolha neste projeto |
|---|---|
| Braços (ofertas) | Canal de contato (`contact`): **cellular** vs **telephone** — proxy real extraído do próprio dataset, sem geração de dados sintéticos |
| Recompensa | Conversão (`y` = assinatura do depósito a prazo), 0/1 |
| Baseline | Regra fixa: sempre recomendar o braço com melhor taxa de conversão histórica |
| Algoritmo adaptativo | Epsilon-Greedy (contextual, modelo de reward por braço) |
| Avaliação | Taxa de conversão, comparação vs. baseline, Golden Set de 5 clientes |
| Serviço | API mínima em FastAPI (`POST /recommend`) que recebe dados de um cliente e retorna a oferta recomendada |
| Nuvem | Arquitetura-alvo descrita em texto (ver seção "Arquitetura-alvo em nuvem" abaixo) |
| MLOps | MLflow local (parâmetros e métricas) |

## Dataset

**UCI Bank Marketing** — arquivo `bank-additional-full.csv`, equivalente ao
dataset Kaggle [`henriqueyamahata/bank-marketing`](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing).

- 41.188 linhas, 20 atributos + variável-alvo `y`.
- A coluna `duration` é removida do conjunto de features: segundo a própria
  documentação do dataset, ela só é conhecida *depois* da ligação
  acontecer, o que causaria vazamento temporal em qualquer modelo preditivo.
- Target desbalanceado: **88,73%** dos clientes não converteram ("no") vs.
  **11,27%** que converteram ("yes"). Por isso, acurácia bruta não é usada
  como métrica — todas as comparações usam taxa de conversão.

Fonte original: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Estrutura do repositório

```
tc5/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # bank-additional-full.csv (não versionado)
│   └── processed/      # arquivos gerados pelos notebooks
├── notebooks/
│   ├── 01_eda.ipynb              # análise exploratória e limpeza
│   ├── 02_preparacao.ipynb       # features, braços e split treino/teste
│   ├── 03_baseline_bandit.ipynb  # baseline, modelo por braço e Epsilon-Greedy
│   └── 04_avaliacao.ipynb        # golden set com oferta recomendada e justificativa
├── models/
│   ├── model_arm0.joblib         # modelo de reward do braço cellular
│   ├── model_arm1.joblib         # modelo de reward do braço telephone
│   └── columns.json              # colunas de X_train, usadas para reindexar a entrada da API
└── src/
    ├── train_policy.py           # treina e salva os dois modelos por braço
    └── api.py                    # API FastAPI (POST /recommend)
```

## Como executar

```bash
git clone <url-deste-repositorio>
cd tc5
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Baixe o dataset `bank-additional-full.csv` (link acima) e coloque em
`data/raw/`. Em seguida, rode os notebooks em ordem, na pasta `notebooks/`:

1. `01_eda.ipynb` — carrega os dados brutos, faz a análise exploratória e
   salva a base limpa em `data/processed/bank_additional_clean.csv`.
2. `02_preparacao.ipynb` — define target/braço, seleciona features,
   aplica one-hot encoding e gera o split treino/teste em
   `data/processed/`.
3. `03_baseline_bandit.ipynb` — calcula a métrica de conversão do
   baseline (regra fixa), treina um modelo de reward (regressão logística)
   por braço, implementa a política Epsilon-Greedy e compara as duas
   abordagens em uma tabela.
4. `04_avaliacao.ipynb` — retreina os modelos por braço, seleciona um
   Golden Set de 5 clientes do teste e gera, para cada um, a oferta
   recomendada com justificativa, salvando o resultado em
   `data/processed/golden_set.csv`.

Depois, para subir o serviço de recomendação (Etapa 5):

```bash
python src/train_policy.py       # treina e salva os modelos em models/ (rodar uma vez)
uvicorn src.api:app --reload     # sobe a API em http://127.0.0.1:8000
```

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "job": "technician", "marital": "married",
    "education": "university.degree", "default": "no", "housing": "yes",
    "loan": "no", "month": "may", "day_of_week": "mon", "campaign": 1,
    "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": 1.1, "cons_price_idx": 93.994, "cons_conf_idx": -36.4,
    "euribor3m": 4.857, "nr_employed": 5191.0
  }'
```

Resposta:

```json
{"oferta_recomendada": "telephone", "probabilidade_cellular": 0.0664, "probabilidade_telephone": 0.0691}
```

## Principais decisões e achados

- **Braços definidos pelo canal de contato**: como o dataset representa uma
  única campanha (não há múltiplas ofertas explícitas), o canal de contato
  (`cellular` vs `telephone`) foi escolhido como proxy dos braços — real,
  não-sintético, e com apenas 2 categorias.
- **Diferença de conversão entre braços confirma o sinal**: 14,74% de
  conversão via `cellular` contra 5,23% via `telephone` — uma diferença
  grande o suficiente para justificar uma política adaptativa em vez de uma
  regra fixa única.
- **18 features de contexto do cliente** foram selecionadas (dados
  demográficos, histórico de contato e indicadores macroeconômicos),
  excluindo o próprio braço (`contact`), o alvo (`y`) e `duration`
  (vazamento).
- **Split treino/teste** feito de forma estratificada pelo alvo (80/20,
  `random_state=42`), preservando a proporção de conversão em ambos os
  conjuntos (11,27% treino / 11,26% teste).
- **Baseline vs. Epsilon-Greedy**: o baseline (sempre `cellular`) atingiu
  14,88% de conversão no teste. O Epsilon-Greedy contextual (modelo de
  reward por braço via regressão logística), avaliado por replay
  (cobertura de ~54% do teste, já que só é possível confirmar o resultado
  quando a escolha da política coincide com o canal realmente usado no
  histórico), não superou o baseline nos epsilons testados — ficou entre
  13,97% (ε=0.10) e 14,21% (ε=0.01), convergindo para perto do baseline
  conforme a exploração diminui. Esse resultado foi mantido e registrado
  como achado honesto, já que a comparação já cumpre o objetivo de mostrar
  o funcionamento e a avaliação de uma política adaptativa frente a uma
  regra fixa.
- **Golden Set de 5 clientes** (amostra aleatória do teste,
  `random_state=42`): para cada cliente, a oferta recomendada foi o braço
  com maior probabilidade prevista pelos dois modelos, com justificativa
  baseada na diferença entre essas probabilidades. Em 2 dos 5 casos a
  recomendação coincidiu com o canal usado no histórico — divergir nos
  outros 3 é esperado, já que a política adaptativa busca melhorar a
  decisão, não repetir o histórico. Nenhum dos 5 clientes converteu de
  fato, o que é consistente com as probabilidades previstas baixas
  (3–19%) e a taxa base de conversão de ~11% da base inteira.
- **Serviço demonstrável (Etapa 5)**: a política treinada foi exposta como
  uma API FastAPI (`POST /recommend`). Os dois modelos por braço são
  treinados uma única vez (`src/train_policy.py`) e salvos em `models/`
  (junto com a lista de colunas usada no treino); a API só carrega esses
  artefatos e monta o vetor de entrada a partir dos dados "crus" do
  cliente, sem precisar retreinar nada a cada chamada. Testada localmente
  com sucesso: para um cliente de exemplo, retornou `telephone` como
  oferta recomendada — divergindo do braço vencedor em média (`cellular`),
  o mesmo comportamento caso a caso já observado no Golden Set da Etapa 4.

## Arquitetura-alvo em nuvem

Este projeto roda hoje inteiramente local (notebooks + API FastAPI localmente.
Numa arquitetura de produção na **AWS**, o fluxo seria:
os dados brutos e processados (equivalentes a `data/raw/` e
`data/processed/`) ficariam em um bucket **S3**, versionado, servindo como
fonte única para os notebooks de treino e para pipelines de re-treino
agendados.

O treino da política (os dois modelos de reward por braço)
rodaria em um job do **SageMaker Training** (ou um container batch
simples), com os artefatos (`model_arm0.joblib`, `model_arm1.joblib`,
`columns.json`) publicados em outro bucket S3 versionado, funcionando como
um registro de modelos mínimo.

Para servir as recomendações, o endpoint FastAPI atual (`src/api.py`) seria
empacotado em um container e implantado como uma função **AWS Lambda** (ou
um **SageMaker Endpoint**, se a latência/escala exigir algo mais robusto),
exposto ao mundo através do **API Gateway** — mantendo a mesma rota
`POST /recommend`.

O **CloudWatch** cuidaria de logs e métricas
operacionais (latência, taxa de erro, volume de chamadas), e também
poderia registrar as decisões do bandit (braço escolhido por chamada) para
alimentar uma futura reavaliação do modelo com dados reais de produção —
fechando o ciclo entre a arquitetura de nuvem e o tracking de MLOps da
Etapa 7.

## Roadmap (etapas do desafio)

- [x] Etapa 0 — Organização do projeto
- [x] Etapa 1 — Base Kaggle e análise exploratória (EDA)
- [x] Etapa 2 — Preparação da base (features, braços, split treino/teste)
- [x] Etapa 3 — Baseline e estratégia algorítmica (Epsilon-Greedy)
- [x] Etapa 4 — Avaliação e Golden Set
- [x] Etapa 5 — Serviço ou interface demonstrável
- [x] Etapa 6 — Arquitetura-alvo em nuvem
- [ ] Etapa 7 — Ciclo de vida MLOps (MLflow)
- [ ] Etapa 8 — Apresentação final (Demo Day)