import logging

import pandas as pd


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Adiciona médias móveis e desvio padrão à série de preços de fechamento.
    Args:
        df (pd.DataFrame): DataFrame original.
        window (int): Janela para cálculo das estatísticas.
    Returns:
        pd.DataFrame: DataFrame com as novas features.
    """
    try:
        df[f"close_ma_{window}"] = df["close"].rolling(window).mean()
        df[f"close_std_{window}"] = df["close"].rolling(window).std()
        logging.info(f"Features de rolling adicionadas (window={window})")
        return df
    except Exception as e:
        logging.error(f"Erro ao adicionar rolling features: {e}")
        raise
