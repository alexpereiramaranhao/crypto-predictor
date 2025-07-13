"""
features.py
-----------
Funções para engenharia de features do projeto Crypto Predictor.

Features adicionadas por padrão:
- Lag do preço de fechamento: close_lag_1, close_lag_2, close_lag_3, close_lag_7
    -> Preço de fechamento de 1, 2, 3 e 7 dias atrás (captura padrões temporais curtos)
- Médias móveis do fechamento: close_mean_7, close_mean_14
    -> Média móvel de 7 e 14 dias do preço de fechamento (captura tendências locais)
- Desvio padrão do fechamento: close_std_7, close_std_14
    -> Volatilidade (dispersão) do preço de fechamento nos últimos 7 e 14 dias
- Mínimo/máximo do fechamento: close_min_7, close_max_7, close_min_14, close_max_14
    -> Mínimo e máximo do preço de fechamento nas janelas de 7 e 14 dias
- Retorno percentual diário: return_1
    -> Variação percentual do fechamento em relação ao dia anterior
- Dia da semana: day_of_week
    -> Valor inteiro de 0 a 6 indicando o dia da semana (pode capturar padrões de calendário)

Todas as features são criadas com base apenas em dados do passado, evitando vazamento de informação.
"""
import pandas as pd

def add_lag_features(df, price_col="close", lags=[1, 2, 3, 7]):
    for lag in lags:
        df[f"{price_col}_lag_{lag}"] = df[price_col].shift(lag)
    return df

def add_rolling_features(df, price_col="close", windows=[7, 14]):
    for window in windows:
        df[f"{price_col}_mean_{window}"] = df[price_col].rolling(window).mean()
        df[f"{price_col}_std_{window}"] = df[price_col].rolling(window).std()
        df[f"{price_col}_min_{window}"] = df[price_col].rolling(window).min()
        df[f"{price_col}_max_{window}"] = df[price_col].rolling(window).max()
    return df

def add_return_features(df, price_col="close"):
    df["return_1"] = df[price_col].pct_change()
    return df

def add_day_of_week(df, date_col="date"):
    df["day_of_week"] = pd.to_datetime(df[date_col]).dt.dayofweek
    return df

def build_features(df, price_col="close", date_col="date"):
    df = df.copy()
    df = add_lag_features(df, price_col)
    df = add_rolling_features(df, price_col)
    df = add_return_features(df, price_col)
    df = add_day_of_week(df, date_col)
    df = df.dropna().reset_index(drop=True)
    return df

