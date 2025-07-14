# Crypto Predictor

Um projeto acadêmico para previsão de preços de criptomoedas usando modelos de aprendizado de máquina.

## 📋 Sobre o Projeto

Este projeto foi desenvolvido para um curso de especialização em Inteligência Artificial. O objetivo é usar modelos básicos de machine learning para prever preços de fechamento de criptomoedas e calcular a lucratividade de diferentes estratégias de investimento.

### O que o projeto faz:

- Carrega dados históricos de **10 criptomoedas** diferentes
- Cria indicadores técnicos básicos (média móvel, volatilidade, retornos)
- Treina modelos de previsão (MLP, Linear, Polinomial)
- Calcula lucros simulados usando as previsões
- Gera **análises estatísticas** com boxplots, histogramas e testes
- Compara variabilidade entre criptomoedas
- Realiza testes de hipóteses e ANOVA

## 🚀 Instalação Rápida

### 1. Clone o repositório

```bash
git clone https://github.com/alexpereiramaranhao/crypto-predictor
cd crypto-predictor
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

ou use Poetry via

```bash
poetry install
```

### 3. Teste se está funcionando

```bash
python -m src.main --model linear
```

## 📁 Estrutura do Projeto

```
crypto-predictor/
├── data/                    # Dados das 10 criptomoedas
│   ├── Poloniex_ADAUSDT_d.csv      # Dados da Cardano
│   ├── Poloniex_BTCUSDT_d.csv      # Dados do Bitcoin
│   └── ...                         # Outras 8 criptomoedas
├── figures/                 # Gráficos gerados
├── src/                     # Código principal
│   ├── data_load.py         # Carregamento de dados
│   ├── features.py          # Criação de indicadores
│   ├── models.py            # Modelos de ML
│   ├── lucro.py             # Cálculos de lucro
│   ├── statistics/          # Análises estatísticas
│   └── util/                # Funções auxiliares
├── tests/                   # Testes automatizados
├── requirements.txt         # Dependências
└── README.md               # Este arquivo
```

## 🔧 Como Usar

### Execução Básica

Para executar o projeto com um modelo específico e processar as 10 criptomoedas:

```bash
python -m src.main --model linear
```

### Parâmetros Disponíveis

#### ⚠️ Parâmetro Obrigatório
- `--model` **(OBRIGATÓRIO)**: Escolha do modelo de previsão
  - `linear`: Regressão linear simples e rápida
  - `mlp`: Rede neural multicamadas (mais complexa)
  - `poly`: Regressão polinomial (captura não-linearidades)

#### 📋 Parâmetros Opcionais
- `--kfolds`: Número de divisões para validação cruzada (padrão: 5, mínimo: 2)
- `--teste-retorno`: Percentual de retorno esperado para teste de hipótese (ex: 5.0 para 5%)
  - Se não especificado, o teste de hipótese não será executado
  - Valor sugerido: entre 1.0 e 10.0 (1% a 10% de retorno)

#### 🆘 Ajuda
Para ver todos os parâmetros disponíveis:
```bash
python -m src.main --help
```

### Exemplos de Uso

```bash
# Usar modelo linear com validação cruzada de 5 folds
python -m src.main --model linear --kfolds 5

# Usar rede neural MLP com 10 folds
python -m src.main --model mlp --kfolds 10

# Usar regressão polinomial
python -m src.main --model poly

# Executar com teste de hipótese para retorno de 3%
python -m src.main --model linear --teste-retorno 3.0
```

### Demonstração Completa

Para ver todas as funcionalidades em ação com todas as 10 criptomoedas:

```bash
python -m src.main --model mlp --kfolds 5 --teste-retorno 2.0
```

⏱️ **Tempo estimado**: 3-5 minutos (processando 10 criptomoedas)

Este comando irá:

1. Carregar dados das 10 criptomoedas
2. Criar indicadores técnicos básicos para todas
3. Treinar modelos de previsão (MLP, Linear, Polinomial)
4. Calcular lucros simulados para cada criptomoeda
5. Realizar análises estatísticas completas (ANOVA, testes de hipótese)
6. Gerar gráficos na pasta `figures/`

### 📊 O que Esperar Durante a Execução

```
🚀 Executando pipeline usando modelo mlp com 5 folds

📊 Processando ADA...
┌─────────────────────────────────────────────────────┐
│            Medidas resumo e de dispersão - ADA     │
├─────────────────────────────────────────────────────┤
│ Estatística                      │ Valor             │
│ mean                            │ 1.234567          │
│ median                          │ 1.123456          │
│ std                             │ 0.456789          │
└─────────────────────────────────────────────────────┘

🔧 Criando features...
🔍 Análise Completa de Lucro - ADA
🤖 Treinando MLP...
📈 Treinando Regressão Linear...
🔢 Encontrando melhor grau polinomial (2-10)...
💰 Calculando lucros...
📊 Gerando gráficos...
✅ ADA processado com sucesso!

[... repete para as outras 9 criptomoedas ...]

📈 RESUMO GERAL
📊 ANÁLISES DE VARIÂNCIA (ANOVA)
🔍 A) ANOVA entre criptomoedas
🔍 B) ANOVA entre grupos de volatilidade
🎉 Pipeline completo executado!
```

## 📊 O que você vai ver

### Indicadores Criados

- **Média Móvel 7 dias**: Suaviza variações de preço
- **Volatilidade**: Medida de risco (desvio padrão)
- **Retorno Diário**: Variação percentual dia a dia
- **Indicador de Alta**: Se o preço subiu (1) ou desceu (0)

### Modelos Treinados

- **Linear**: Simples e rápido
- **MLP**: Rede neural básica
- **Polinomial**: Captura relações não-lineares

### Análises Geradas

#### 📊 Para cada criptomoeda:

- **Estatísticas descritivas**: média, mediana, moda, desvio padrão, variância
- **Boxplots**: distribuição dos preços de fechamento
- **Histogramas**: frequência dos preços com curva de densidade
- **Gráficos de linha**: preços + média móvel, mediana móvel e moda

#### 📈 Análises comparativas:

- **Variabilidade entre criptomoedas**: comparação de dispersão
- **Teste de hipóteses**: retornos médios superiores a valor esperado
- **ANOVA entre criptomoedas**: diferenças significativas nos retornos
- **ANOVA por grupos**: agrupamento por volatilidade com testes post-hoc
- **Validação cruzada**: performance dos modelos
- **Comparação de lucros**: estratégias de investimento

## 🧪 Executando Testes

### Testes Básicos

```bash
pytest
```

### Testes com Relatório de Cobertura

```bash
pytest --cov=src --cov-report=html
```

Depois abra `htmlcov/index.html` no navegador para ver o relatório.

### Testes de Arquivo Específico

```bash
pytest tests/test_features.py
```

## 🔍 Qualidade de Código

### Formatação Automática

```bash
black src/ tests/
```

### Verificação de Estilo

```bash
ruff check --fix src/ tests/
```

### Verificação de Segurança

```bash
bandit -r src/
safety scan
```

## 📈 Exemplos de Saída

### Processamento Multi-Criptomoedas

```
Executando pipeline para múltiplas moedas usando modelo linear com 5 folds.
Processando ADA: 1000 dias de dados
Processando AKITA: 850 dias de dados
Processando BTC: 1200 dias de dados
...
```

### Análises Estatísticas Geradas

```
┌─────────────────────────────────────────────────────┐
│            Medidas resumo e de dispersão - BTC      │
├─────────────────────────────────────────────────────┤
│ Estatística                      │ Valor             │
│ mean                            │ 45123.456789      │
│ median                          │ 43500.000000      │
│ std                             │ 12345.678901      │
└─────────────────────────────────────────────────────┘
```

### 📁 Arquivos Gerados

Após a execução, você encontrará em `figures/`:

#### 📊 **Por criptomoeda** (30 arquivos - 3 × 10 criptos):
- **Boxplots**: `boxplot_BTC.png`, `boxplot_ADA.png`, etc.
- **Histogramas**: `histogram_BTC.png`, `histogram_ADA.png`, etc.  
- **Gráficos de linha**: `price_summary_BTC.png`, etc.

#### 🔄 **Comparações de modelos** (2 arquivos):
- **Evolução do lucro**: `evolucao_lucro_modelos.png` (subplots 1x3)
- **Dispersão de previsões**: `dispersao_modelos.png` (subplots 1x3)

🎯 **Total**: ~32 arquivos PNG (resolução 150 DPI)

## 🔧 Solução de Problemas

### ❌ Erros Comuns

#### 1. **"argument --model is required"**
```bash
# ❌ Erro
python -m src.main

# ✅ Correto  
python -m src.main --model linear
```

#### 2. **"No such file or directory: data/Poloniex_..."**
- Verifique se está na pasta raiz do projeto
- Confirme se a pasta `data/` existe com os arquivos CSV

#### 3. **"ModuleNotFoundError: No module named 'src'"**
```bash
# ❌ Erro - executando de pasta errada
cd src
python main.py

# ✅ Correto - executar da pasta raiz
cd crypto-predictor  
python -m src.main --model linear
```

#### 4. **Execução muito lenta**
- Normal: processa 10 criptomoedas com 3 modelos cada
- Use menos folds: `--kfolds 3` (em vez de 5)
- Teste com modelo mais rápido: `--model linear`

### 💡 Dicas de Performance

- **Primeiro teste**: `python -m src.main --model linear --kfolds 3`
- **Análise completa**: `python -m src.main --model mlp --kfolds 5`
- **Com testes de hipótese**: adicione `--teste-retorno 2.0`

## ⚠️ Limitações Atuais

### Dados

- ✅ 10 criptomoedas já incluídas no repositório
- Dados obtidos de [CryptoDataDownload](https://www.cryptodatadownload.com)
- Formato padronizado USDT para facilitar comparações

### Modelos

- Apenas 3 modelos básicos implementados
- Não há otimização automática de hiperparâmetros

### Features

- 4 indicadores técnicos básicos
- Não há RSI, MACD ou outros indicadores avançados

## 🎯 Requisitos Acadêmicos Atendidos

Este projeto atende aos requisitos do trabalho acadêmico:

- ✅ Modelos de ML para previsão de preços
- ✅ Validação cruzada K-fold
- ✅ Análise estatística para 10 criptomoedas
- ✅ Medidas resumo e dispersão completas
- ✅ Boxplots e histogramas individuais
- ✅ Análise de variabilidade entre criptomoedas
- ✅ Gráficos de linha com média, mediana e moda
- ✅ Teste de hipóteses e ANOVA
- ✅ Cálculo de lucro com estratégia de investimento
- ✅ Comparação entre modelos
- ✅ Visualizações com matplotlib/seaborn
- ✅ Estrutura modular com docstrings
- ✅ Testes automatizados com pytest
- ✅ Ferramentas de qualidade de código

## 🤝 Contribuindo

Este é um projeto acadêmico, mas se você quiser contribuir:

1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Rode os testes e formatação
4. Faça um pull request

## 📧 Contato

Para dúvidas sobre o projeto acadêmico:

- noronha@ifg.edu.br
- otavio.xavier@ifg.edu.br
- eder.brito@ifg.edu.br

Para dúvidas sobre a implementação:

- eduardocbraga@hotmail.com
- alexpereiramaranhao@outlook.com

---

**Nota**: Este é um projeto educacional para fins acadêmicos. Não deve ser usado para decisões reais de investimento.
