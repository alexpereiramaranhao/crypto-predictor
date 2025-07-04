import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
from typing import Dict, List

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

def summary_statistics(df: pd.DataFrame, price_col: str = "close") -> Dict[str, float]:
    """
    Retorna medidas resumo e de dispersão do preço de fechamento.
    """
    desc = df[price_col].describe()
    stats = {
        "mean": desc["mean"],
        "median": df[price_col].median(),
        "mode": df[price_col].mode().iloc[0] if not df[price_col].mode().empty else np.nan,
        "min": desc["min"],
        "max": desc["max"],
        "std": desc["std"],
        "var": df[price_col].var(),
        "25%": desc["25%"],
        "50%": desc["50%"],
        "75%": desc["75%"],
    }

    return stats

def compare_dispersion(dfs: Dict[str, pd.DataFrame], price_col: str = "close") -> pd.DataFrame:
    """
    Recebe um dicionário {nome: DataFrame} e retorna tabela com std e var de cada moeda.
    """
    data = []
    for crypto, df in dfs.items():
        std = df[price_col].std()
        var = df[price_col].var()
        data.append({"crypto": crypto, "std": std, "var": var})
    result = pd.DataFrame(data)
    logging.info(f"Comparação de dispersão:\n{result}")
    return result
