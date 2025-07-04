# Crypto Forecasting

Projeto de previsão de preços de criptomoedas utilizando modelos de regressão (MLP, linear, polinomial, etc).

## Como usar

1. Clone este repositório e instale as dependências:
    ```
    pip install -r requirements.txt
    ```
2. Execute o script principal:
    ```
    python main.py --crypto BTC --model mlp --kfolds 5
    ```
3. Os gráficos serão salvos na pasta `figures/`.

## Estrutura

```text
crypto-predictor/
│
├── data/                # Para datasets locais ou temporários
│
├── figures/             # Salvar todos os gráficos gerados
│
├── src/                 # Código-fonte principal do projeto
│   ├── __init__.py
│   ├── data_load.py     # Funções de carregamento dos dados
│   ├── features.py      # Engenharia de features
│   ├── models.py        # Definição/treinamento de modelos
│   ├── analysis.py      # Análises estatísticas (medidas resumo, ANOVA, etc)
│   ├── plots.py         # Funções para gerar e salvar gráficos
│   ├── utils.py         # Funções auxiliares (ex: cálculo de lucro, logging customizado)
│   └── config.py        # Configurações centralizadas, ex: lista de moedas, paths
│
├── tests/               # Testes unitários com pytest
│   ├── __init__.py
│   ├── test_data_load.py
│   ├── test_features.py
│   └── test_models.py
│
├── requirements.txt     # Todas as dependências do projeto
├── README.md            # Documentação e instruções de uso
├── main.py              # Script principal para execução via CLI
└── .gitignore           # Para ignorar arquivos/diretórios no controle de versão (git)
```