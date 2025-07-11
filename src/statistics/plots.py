import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

def plot_boxplot(df: pd.DataFrame, price_col: str = "close", crypto: str = "BTC"):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[price_col])
    plt.title(f"Boxplot do preço de fechamento - {crypto}")
    plt.xlabel("Preço de Fechamento")
    plt.tight_layout()
    plt.savefig(f"figures/boxplot_{crypto}.png", dpi=150)
    plt.close()

def plot_histogram(df: pd.DataFrame, price_col: str = "close", crypto: str = "BTC"):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[price_col], kde=True, bins=30)
    plt.title(f"Histograma do preço de fechamento - {crypto}")
    plt.xlabel("Preço de Fechamento")
    plt.tight_layout()
    plt.savefig(f"figures/histogram_{crypto}.png", dpi=150)
    plt.close()

def plot_price_with_summary(df: pd.DataFrame, date_col: str = "date", price_col: str = "close", crypto: str = "BTC"):
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[price_col], label="Fechamento")
    plt.plot(df[date_col], df[price_col].rolling(7).mean(), label="Média Móvel (7d)", linestyle="--")
    plt.plot(df[date_col], df[price_col].rolling(7).median(), label="Mediana Móvel (7d)", linestyle=":")
    moda = df[price_col].mode()[0]
    plt.axhline(y=moda, color='r', linestyle='-.', label=f"Moda: {moda:.2f}")
    plt.title(f"{crypto}: Preço de fechamento, média, mediana e moda ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/price_summary_{crypto}.png", dpi=150)
    plt.close()

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
    return result