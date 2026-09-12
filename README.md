# TC5 — Datathon MLET (contexto consolidado)

Este documento resume o estado do projeto para retomar o trabalho em uma nova
conversa. Detalhes completos do plano ficam em `claude/backlog-plan.md`.

## O desafio

Construir uma plataforma mínima de experimentação adaptativa (multi-armed
bandit) que decide qual oferta apresentar a um cliente de banco, com:
baseline determinístico, algoritmo adaptativo (Epsilon-Greedy), avaliação
com golden set, um serviço/demo mínimo, tracking MLOps (MLflow) e um
parágrafo de arquitetura em nuvem. Fonte: `POSTECH - MLET - DATATHON.pdf`
(arquivo do projeto).

## Estrutura do repositório (`tc5/`)

```
tc5/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # bank-full.csv, bank.csv, bank-additional-full.csv,
│   │                   # bank-additional.csv, *-names.txt
│   └── processed/
│       └── bank_additional_clean.csv   # criado na Etapa 1
├── notebooks/
│   └── 01_eda.ipynb    # criado na Etapa 1 (4 células)
└── src/
    └── __init__.py
```

## Dataset escolhido

UCI Bank Marketing — arquivo `bank-additional-full.csv` (41.188 linhas, 20
atributos + target `y`), equivalente ao dataset Kaggle
`henriqueyamahata/bank-marketing` (decisão já registrada no backlog).

## Etapa 1 — Base Kaggle e EDA (Versão incial)

Notebook `notebooks/01_eda.ipynb`, possui o estudo do dataset:

1. **Load**: `pd.read_csv("../data/raw/bank-additional-full.csv", sep=";")`
   → confirmado shape `(41188, 21)`.
2. **Tipos e nulos**: colunas categóricas (`object`) vs numéricas confirmadas
   como esperado. Valores `"unknown"` por coluna: `default` 8.597 (~21% —
   proporção alta, vale considerar ao escolher features), `education` 1.731,
   `housing`/`loan` 990 cada, `job` 330, `marital` 80.
3. **Distribuição do target**: `y` = "no" 36.548 (88,73%) / "yes" 4.640
   (11,27%) — base bem desbalanceada. Implicação registrada: acurácia bruta
   não é métrica útil; comparações futuras (baseline vs. Epsilon-Greedy)
   devem usar taxa de conversão.
4. **Remoção de `duration`** (vazamento temporal, confirmado nos docs do
   UCI) + salvar `data/processed/bank_additional_clean.csv` (shape final
   `(41188, 20)`). Esse valor não é relevante como feature, afinal só o temos
   ao fim da ligação, sendo assim não contribui para fins preditivos.

## Próximo passo: Etapa 2 — Preparação da base

- Selecionar features do cliente + variável alvo.
- Definir os braços (ofertas) a partir dos dados existentes (sem gerar
  dados sintéticos complexos) — usar `data/processed/bank_additional_clean.csv`
  como ponto de partida.
- Split treino/teste simples.