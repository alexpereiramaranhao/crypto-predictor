import logging
import json
import pandas as pd


def load_crypto_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset de uma criptomoeda a partir de um arquivo CSV.
    Args:
        filepath (str): Caminho para o arquivo CSV.
    Returns:
        pd.DataFrame: DataFrame com os dados da criptomoeda.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Arquivo {filepath} carregado com sucesso!")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo {filepath}: {e}")
        raise
