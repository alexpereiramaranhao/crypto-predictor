import logging
from typing import Dict

import numpy as np
import pandas as pd
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)


def summary_statistics(df: pd.DataFrame, price_col: str = "close") -> Dict[str, float]:
    """
    Retorna medidas resumo e de dispersão do preço de fechamento.
    """
    desc = df[price_col].describe()

    q1 = desc["25%"]
    q3 = desc["75%"]

    stats = {
        "mean": desc["mean"],
        "median": df[price_col].median(),
        "mode": (
            df[price_col].mode().iloc[0] if not df[price_col].mode().empty else np.nan
        ),
        "min": desc["min"],
        "max": desc["max"],
        "std": desc["std"],
        "var": df[price_col].var(),
        "amplitude": desc["max"] - desc["min"],
        "iqr": q3 - q1,
        "25%": desc["25%"],
        "50%": desc["50%"],
        "75%": desc["75%"],
    }

    return stats


def compare_dispersion(
    dfs: Dict[str, pd.DataFrame], price_col: str = "close"
) -> pd.DataFrame:
    """
    Compara a variabilidade (dispersão) do preço de fechamento entre criptomoedas.
    Retorna um DataFrame com std, var, amplitude e IQR de cada cripto.
    Args:
        dfs (dict): Dicionário {nome: DataFrame}
        price_col (str): Nome da coluna de preços
    Returns:
        pd.DataFrame: Medidas de dispersão por moeda
    """
    data = []
    for crypto, df in dfs.items():
        desc = df[price_col].describe()
        q1 = desc["25%"]
        q3 = desc["75%"]
        row = {
            "crypto": crypto,
            "std": df[price_col].std(),
            "var": df[price_col].var(),
            "amplitude": desc["max"] - desc["min"],
            "iqr": q3 - q1,
        }
        data.append(row)
    result = pd.DataFrame(data)
    return result
