# Crypto Predictor

Project for cryptocurrency price prediction using regression models (MLP, linear, polynomial, etc.).

## Setup and Installation

1. Clone this repository:
    ``` bash
    git clone https://github.com/alexpereiramaranhao/crypto-predictor
    ```
2. Install dependencies:
    ``` bash
    pip install -r requirements.txt
    ```

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
│   ├── test_analysis.py
│
├── requirements.txt     # Todas as dependências do projeto
├── README.md            # Documentação e instruções de uso
├── main.py              # Script principal para execução via CLI
└── .gitignore           # Para ignorar arquivos/diretórios no controle de versão (git)
```
## How to Use

### Basic Usage

Run the main script with the desired cryptocurrency and model in project's root directory:

``` bash
python main.py --crypto BTC --model linear --kfolds 5
```

Run the main script with all cryptocurrency:

``` bash
python -m src.main --model linear
```

### Available Parameters

- `--crypto`: Cryptocurrency symbol (default: BTC)
   - Available options: BTC, ETH, ADA, etc.
- `--model`: Model to use (default: mlp)
   - Available options: mlp, linear, polynomial, random_forest, xgboost
- `--kfolds`: Number of k-fold cross-validation splits (default: 5)
- `--timeframe`: Data timeframe in days (default: 365)
- `--features`: Comma-separated list of features to use (default: "close,volume,rsi,macd")

### Examples

Train a polynomial regression model on Ethereum with 10-fold cross-validation:

``` bash
python -m src.main --crypto ETH --model polynomial --kfolds 10
```

Use specific features with a random forest model:

``` bash
python -m src.main --crypto BTC --model random_forest --features "close,volume,rsi,ma20,ma50"
```

## Running Tests

### Basic Tests

Run the test suite with pytest:

``` bash
pytest
```

### Tests with Coverage

Run tests with coverage report:

``` bash
pytest --cov=src --cov-report=html
```
Then open `htmlcov/index.html` in your browser to view the report.

## Output
Results will be saved to the `figures/` directory, including:
* Charts