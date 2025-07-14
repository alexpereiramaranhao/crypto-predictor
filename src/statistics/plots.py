from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_price_with_summary(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "close",
    crypto: str = "BTC",
):
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[price_col], label="Fechamento")
    plt.plot(
        df[date_col],
        df[price_col].rolling(7).mean(),
        label="Média Móvel (7d)",
        linestyle="--",
    )
    plt.plot(
        df[date_col],
        df[price_col].rolling(7).median(),
        label="Mediana Móvel (7d)",
        linestyle=":",
    )
    moda = df[price_col].mode()[0]
    plt.axhline(y=moda, color="r", linestyle="-.", label=f"Moda: {moda:.2f}")
    plt.title(f"{crypto}: Preço de fechamento, média, mediana e moda ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/price_summary_{crypto}.png", dpi=150)
    plt.close()


def compare_dispersion(
    dfs: Dict[str, pd.DataFrame], price_col: str = "close"
) -> pd.DataFrame:
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


def plotar_evolucao_lucro(historico_dinheiro_modelo1, historico_dinheiro_modelo2, 
                                  nome_modelo1="Modelo 1", nome_modelo2="Modelo 2"):
    """
    Plota a evolução do lucro de dois modelos ao longo do tempo.
    
    Args:
        historico_dinheiro_modelo1: Lista com evolução do dinheiro do modelo 1
        historico_dinheiro_modelo2: Lista com evolução do dinheiro do modelo 2
        nome_modelo1: Nome do primeiro modelo
        nome_modelo2: Nome do segundo modelo
    """
    print("Criando gráfico de evolução do lucro...")
    
    plt.figure(figsize=(12, 6))
    
    # Dias (eixo X)
    dias = list(range(len(historico_dinheiro_modelo1)))
    
    # Plota as duas linhas
    plt.plot(dias, historico_dinheiro_modelo1, label=nome_modelo1, linewidth=2, marker='o', markersize=4)
    plt.plot(dias, historico_dinheiro_modelo2, label=nome_modelo2, linewidth=2, marker='s', markersize=4)
    
    # Adiciona linha horizontal no investimento inicial
    investimento_inicial = historico_dinheiro_modelo1[0]
    plt.axhline(y=investimento_inicial, color='gray', linestyle='--', alpha=0.7, 
                label=f'Investimento inicial (R$ {investimento_inicial:.2f})')
    
    plt.title("Evolução do Lucro dos Modelos ao Longo do Tempo")
    plt.xlabel("Dias")
    plt.ylabel("Valor da Carteira (R$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/evolucao_lucro_modelos.png", dpi=150)
    plt.close()
    
    print("Gráfico salvo em figures/evolucao_lucro_modelos.png")


def plotar_dispersao_modelos(precos_reais, previsoes_modelo1, previsoes_modelo2,
                                     nome_modelo1="Modelo 1", nome_modelo2="Modelo 2"):
    """
    Cria gráfico de dispersão comparando previsões vs preços reais.
    
    Args:
        precos_reais: Preços reais da criptomoeda
        previsoes_modelo1: Previsões do modelo 1
        previsoes_modelo2: Previsões do modelo 2
    """
    print("Criando gráfico de dispersão dos modelos...")
    
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Modelo 1
    plt.subplot(1, 3, 1)
    plt.scatter(precos_reais, previsoes_modelo1, alpha=0.6, color='blue')
    plt.plot([min(precos_reais), max(precos_reais)], [min(precos_reais), max(precos_reais)], 
             'r--', label='Previsão perfeita')
    plt.xlabel("Preços Reais")
    plt.ylabel("Previsões")
    plt.title(f"Dispersão - {nome_modelo1}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Modelo 2
    plt.subplot(1, 3, 2)
    plt.scatter(precos_reais, previsoes_modelo2, alpha=0.6, color='green')
    plt.plot([min(precos_reais), max(precos_reais)], [min(precos_reais), max(precos_reais)], 
             'r--', label='Previsão perfeita')
    plt.xlabel("Preços Reais")
    plt.ylabel("Previsões")
    plt.title(f"Dispersão - {nome_modelo2}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparação lado a lado
    plt.subplot(1, 3, 3)
    plt.scatter(precos_reais, previsoes_modelo1, alpha=0.6, color='blue', label=nome_modelo1)
    plt.scatter(precos_reais, previsoes_modelo2, alpha=0.6, color='green', label=nome_modelo2)
    plt.plot([min(precos_reais), max(precos_reais)], [min(precos_reais), max(precos_reais)], 
             'r--', label='Previsão perfeita')
    plt.xlabel("Preços Reais")
    plt.ylabel("Previsões")
    plt.title("Comparação dos Modelos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/dispersao_modelos.png", dpi=150)
    plt.close()
    
    print("Gráfico salvo em figures/dispersao_modelos.png")


def plotar_retornos_entre_criptos(dados_criptos_dict):
    """
    Plota boxplot dos retornos diários de várias criptomoedas para comparação.
    
    Args:
        dados_criptos_dict: Dicionário com {nome_crypto: dataframe}
    """
    print("Criando gráfico de comparação de retornos entre criptomoedas...")
    
    # Prepara os dados
    dados_para_plot = []
    for nome_crypto, dataframe in dados_criptos_dict.items():
        if 'retorno_diario' in dataframe.columns:
            retornos_limpos = dataframe['retorno_diario'].dropna()
            for retorno in retornos_limpos:
                dados_para_plot.append({'Criptomoeda': nome_crypto, 'Retorno_Diario': retorno})
    
    df_plot = pd.DataFrame(dados_para_plot)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='Criptomoeda', y='Retorno_Diario')
    plt.title("Comparação dos Retornos Diários entre Criptomoedas")
    plt.xlabel("Criptomoeda")
    plt.ylabel("Retorno Diário (%)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/comparacao_retornos_criptos.png", dpi=150)
    plt.close()
    
    print("Gráfico salvo em figures/comparacao_retornos_criptos.png")
