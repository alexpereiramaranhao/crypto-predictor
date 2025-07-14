import logging
from typing import Union

import numpy as np
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


def calcular_media_movel(dados_preco: Union[pd.Series, list], janela_dias: int = 7) -> pd.Series:
    """
    Calcula a média móvel dos preços.
    
    Args:
        dados_preco: Lista ou Series com os preços
        janela_dias: Quantos dias usar para calcular a média
    
    Returns:
        media_movel: Lista com as médias móveis
    """
    print(f"Calculando média móvel de {janela_dias} dias...")
    
    # Converte para pandas Series se não for
    if not isinstance(dados_preco, pd.Series):
        dados_preco = pd.Series(dados_preco)
    
    # Calcula a média móvel
    media_movel = dados_preco.rolling(window=janela_dias).mean()
    
    print(f"Média móvel calculada! {len(media_movel)} valores gerados.")
    return media_movel


def calcular_volatilidade(dados_preco: Union[pd.Series, list], janela_dias: int = 7) -> pd.Series:
    """
    Calcula a volatilidade (desvio padrão) dos preços.
    
    Args:
        dados_preco: Lista ou Series com os preços
        janela_dias: Quantos dias usar para calcular a volatilidade
    
    Returns:
        volatilidade: Lista com as volatilidades
    """
    print(f"Calculando volatilidade de {janela_dias} dias...")
    
    # Converte para pandas Series se não for
    if not isinstance(dados_preco, pd.Series):
        dados_preco = pd.Series(dados_preco)
    
    # Calcula a volatilidade (desvio padrão)
    volatilidade = dados_preco.rolling(window=janela_dias).std()
    
    print(f"Volatilidade calculada! {len(volatilidade)} valores gerados.")
    return volatilidade


def calcular_retorno(dados_preco: Union[pd.Series, list]) -> pd.Series:
    """
    Calcula o retorno entre dias consecutivos.
    
    Args:
        dados_preco: Lista ou Series com os preços
    
    Returns:
        retornos: Lista com os retornos percentuais
    """
    print("Calculando retornos...")
    
    # Converte para pandas Series se não for
    if not isinstance(dados_preco, pd.Series):
        dados_preco = pd.Series(dados_preco)
    
    # Calcula os retornos: (preço_hoje - preço_ontem) / preço_ontem * 100
    retornos = dados_preco.pct_change() * 100
    
    print(f"Retornos calculados! {len(retornos)} valores gerados.")
    return retornos


def criar_features_basicas_completas(dataframe_crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de uma vez para um DataFrame de criptomoeda.
    
    Args:
        dataframe_crypto: DataFrame com dados da crypto (deve ter coluna 'close')
    
    Returns:
        dataframe_com_features: DataFrame original + novas features
    """
    print("Criando features...")
    
    # Faz uma cópia para não modificar o original
    df_resultado = dataframe_crypto.copy()
    
    # Adiciona média móvel de 7 dias
    df_resultado['media_movel_7d'] = calcular_media_movel(df_resultado['close'], 7)
    
    # Adiciona volatilidade de 7 dias
    df_resultado['volatilidade_7d'] = calcular_volatilidade(df_resultado['close'], 7)
    
    # Adiciona retornos diários
    df_resultado['retorno_diario'] = calcular_retorno(df_resultado['close'])
    
    # Adiciona se o preço subiu ou desceu (1 = subiu, 0 = desceu)
    df_resultado['preco_subiu'] = (df_resultado['retorno_diario'] > 0).astype(int)
    
    print(f"Features adicionadas: media_movel_7d, volatilidade_7d, retorno_diario, preco_subiu")
    
    return df_resultado
