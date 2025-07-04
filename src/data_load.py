import pandas as pd
import logging
from typing import Optional

def load_crypto_data(filepath: str, sep: str = ",", parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Carrega um dataset de criptomoeda a partir de um arquivo CSV.

    Args:
        filepath (str): Caminho para o arquivo CSV.
        sep (str, opcional): Separador de campo do arquivo CSV. Padrão é ','.
        parse_dates (list, opcional): Lista de colunas a serem interpretadas como datas.

    Returns:
        pd.DataFrame: DataFrame com os dados da criptomoeda.
    """
    try:
        logging.info(f"Carregando arquivo: {filepath}")
        df = pd.read_csv(filepath, sep=sep, parse_dates=parse_dates)
        logging.info(f"Arquivo carregado com sucesso! {df.shape[0]} linhas, {df.shape[1]} colunas.")
        # Remove possíveis colunas irrelevantes (exemplo: índice vindo do CSV)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo {filepath}: {e}")
        raise
