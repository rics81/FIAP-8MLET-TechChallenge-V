# TC5 — Plataforma de Experimentação Adaptativa de Ofertas

Projeto desenvolvido para o **Datathon da POSTECH/FIAP — MLET (Machine
Learning Engineering)**. O desafio propõe construir, de ponta a ponta, uma
plataforma de experimentação adaptativa que decide qual oferta apresentar a
um cliente de uma instituição financeira digital, no lugar de regras fixas
ou testes A/B longos.

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
| Serviço | Script Python / API mínima (FastAPI) que recebe dados de um cliente e retorna a oferta recomendada |
| MLOps | MLflow local (parâmetros e métricas) |
| Nuvem | Arquitetura-alvo descrita em texto (ver seção abaixo, a preencher na Etapa 6) |

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
│   └── 03_baseline_bandit.ipynb  # baseline, modelo por braço e Epsilon-Greedy
└── src/
```

## Como executar

```bash
git clone https://github.com/rics81/FIAP-8MLET-TechChallenge-V
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

(Os notebooks seguintes serão adicionados conforme o projeto avança — ver
roadmap abaixo.)

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

## Roadmap (etapas do desafio)

- [x] Etapa 0 — Organização do projeto
- [x] Etapa 1 — Base Kaggle e análise exploratória (EDA)
- [x] Etapa 2 — Preparação da base (features, braços, split treino/teste)
- [x] Etapa 3 — Baseline e estratégia algorítmica (Epsilon-Greedy)
- [ ] Etapa 4 — Avaliação e Golden Set
- [ ] Etapa 5 — Serviço ou interface demonstrável
- [ ] Etapa 6 — Arquitetura-alvo em nuvem
- [ ] Etapa 7 — Ciclo de vida MLOps (MLflow)
- [ ] Etapa 8 — Apresentação final (Demo Day)

## Considerações éticas e de dados

Este projeto usa exclusivamente uma base pública e anonimizada (UCI/Kaggle
Bank Marketing), sem dados reais de clientes, identificadores, patrimônio,
renda ou atributos sensíveis (gênero, raça). Decisões de oferta permanecem
como recomendação — não há automação de decisões sensíveis sem revisão
humana.