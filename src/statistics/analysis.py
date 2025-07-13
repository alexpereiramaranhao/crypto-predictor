import pandas as pd
import numpy as np
import logging
from rich.logging import RichHandler
from typing import Dict
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

    q1 = desc["25%"]
    q3 = desc["75%"]

    stats = {
        "mean": desc["mean"],
        "median": df[price_col].median(),
        "mode": df[price_col].mode().iloc[0] if not df[price_col].mode().empty else np.nan,
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

def compare_dispersion(dfs: Dict[str, pd.DataFrame], price_col: str = "close") -> pd.DataFrame:
    """
    Compara a variabilidade (dispersão) do preço de fechamento entre criptomoedas.
    Retorna um DataFrame com std, var, amplitude e IQR (intervalo interquartil) de cada cripto.
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
        std = df[price_col].std()
        row = {
            "crypto": crypto,
            "std": std,
            "var": df[price_col].var(),
            "amplitude": desc["max"] - desc["min"],
            "iqr": q3 - q1,
            "coef. variação": std/df[price_col].mean()
        }
        data.append(row)
    result = pd.DataFrame(data)
    return result

def anova_between_cryptos(returns_dict: dict):
    """
    Realiza ANOVA entre os retornos diários das criptomoedas.
    returns_dict: {nome_moeda: array/pd.Series de retornos}
    """
    labels = []
    groups = []
    for name, series in returns_dict.items():
        labels.extend([name]*len(series))
        groups.extend(series)
    # Para ANOVA
    f_stat, p_value = f_oneway(*[pd.Series(r) for r in returns_dict.values()])
    return f_stat, p_value, labels, groups

def tukey_posthoc(returns_dict: dict):
    """
    Teste post hoc de Tukey HSD entre as criptomoedas, após ANOVA.
    """
    labels = []
    groups = []
    for name, series in returns_dict.items():
        labels.extend([name]*len(series))
        groups.extend(series)
    tukey = pairwise_tukeyhsd(endog=groups, groups=labels, alpha=0.05)
    return tukey