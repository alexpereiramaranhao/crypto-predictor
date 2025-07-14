import logging
from typing import Dict

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from scipy import stats

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


def teste_hipotese_retorno(retornos_diarios, percentual_esperado=5.0, nivel_significancia=0.05):
    """
    Teste de hipótese para verificar se o retorno médio é maior que um valor esperado.
    
    Hipótese nula (H0): retorno médio <= percentual_esperado
    Hipótese alternativa (H1): retorno médio > percentual_esperado
    
    Args:
        retornos_diarios: Lista ou Series com os retornos diários em %
        percentual_esperado: Percentual de retorno esperado (ex: 5.0 para 5%)
        nivel_significancia: Nível de significância (padrão: 0.05 para 5%)
    
    Returns:
        resultado: Dicionário com os resultados do teste
    """
    print(f"Realizando teste de hipótese...")
    print(f"H0: retorno médio <= {percentual_esperado}%")
    print(f"H1: retorno médio > {percentual_esperado}%")
    print(f"Nível de significância: {nivel_significancia * 100}%")
    
    # Remove valores NaN se houver
    retornos_limpos = pd.Series(retornos_diarios).dropna()
    
    # Calcula estatísticas básicas
    retorno_medio = retornos_limpos.mean()
    desvio_padrao = retornos_limpos.std()
    tamanho_amostra = len(retornos_limpos)
    
    print(f"Retorno médio da amostra: {retorno_medio:.4f}%")
    print(f"Desvio padrão: {desvio_padrao:.4f}%")
    print(f"Tamanho da amostra: {tamanho_amostra}")
    
    # Calcula estatística t
    estatistica_t = (retorno_medio - percentual_esperado) / (desvio_padrao / np.sqrt(tamanho_amostra))
    
    # Calcula p-valor (teste unilateral à direita)
    graus_liberdade = tamanho_amostra - 1
    p_valor = 1 - stats.t.cdf(estatistica_t, graus_liberdade)
    
    # Decide sobre a hipótese
    rejeita_h0 = p_valor < nivel_significancia
    
    print(f"\nEstatística t: {estatistica_t:.4f}")
    print(f"Graus de liberdade: {graus_liberdade}")
    print(f"P-valor: {p_valor:.6f}")
    
    if rejeita_h0:
        print(f"RESULTADO: Rejeitamos H0 (p < {nivel_significancia})")
        print(f"CONCLUSÃO: O retorno médio É SIGNIFICATIVAMENTE MAIOR que {percentual_esperado}%")
    else:
        print(f"RESULTADO: Não rejeitamos H0 (p >= {nivel_significancia})")
        print(f"CONCLUSÃO: Não há evidências de que o retorno médio seja maior que {percentual_esperado}%")
    
    resultado = {
        "retorno_medio": retorno_medio,
        "percentual_esperado": percentual_esperado,
        "estatistica_t": estatistica_t,
        "p_valor": p_valor,
        "rejeita_h0": rejeita_h0,
        "nivel_significancia": nivel_significancia,
        "tamanho_amostra": tamanho_amostra
    }
    
    return resultado


def anova_entre_criptos(dados_criptos_dict):
    """
    ANOVA para comparar retornos médios entre diferentes criptomoedas.
    
    Args:
        dados_criptos_dict: Dicionário com {nome_crypto: dataframe}
                           Cada dataframe deve ter coluna 'retorno_diario'
    
    Returns:
        resultado: Dicionário com os resultados da ANOVA
    """
    print("Realizando ANOVA entre criptomoedas...")
    
    # Prepara os dados para ANOVA
    grupos_retornos = []
    nomes_criptos = []
    
    for nome_crypto, dataframe in dados_criptos_dict.items():
        if 'retorno_diario' in dataframe.columns:
            retornos_limpos = dataframe['retorno_diario'].dropna()
            if len(retornos_limpos) > 0:
                grupos_retornos.append(retornos_limpos.tolist())
                nomes_criptos.append(nome_crypto)
                print(f"{nome_crypto}: {len(retornos_limpos)} observações, média {retornos_limpos.mean():.4f}%")
    
    if len(grupos_retornos) < 2:
        print("ERRO: Precisa de pelo menos 2 grupos para fazer ANOVA!")
        return None
    
    # Realiza ANOVA
    estatistica_f, p_valor = stats.f_oneway(*grupos_retornos)
    
    # Calcula médias de cada grupo
    medias_grupos = [np.mean(grupo) for grupo in grupos_retornos]
    
    print(f"\nEstatística F: {estatistica_f:.4f}")
    print(f"P-valor: {p_valor:.6f}")
    
    nivel_significancia = 0.05
    if p_valor < nivel_significancia:
        print(f"RESULTADO: Rejeitamos H0 (p < {nivel_significancia})")
        print("CONCLUSÃO: Há diferenças significativas entre os retornos médios das criptomoedas")
        
        # Mostra qual tem maior retorno
        indice_maior = np.argmax(medias_grupos)
        crypto_melhor = nomes_criptos[indice_maior]
        print(f"Criptomoeda com maior retorno médio: {crypto_melhor} ({medias_grupos[indice_maior]:.4f}%)")
    else:
        print(f"RESULTADO: Não rejeitamos H0 (p >= {nivel_significancia})")
        print("CONCLUSÃO: Não há diferenças significativas entre os retornos médios")
    
    resultado = {
        "estatistica_f": estatistica_f,
        "p_valor": p_valor,
        "grupos": nomes_criptos,
        "medias_grupos": medias_grupos,
        "diferencas_significativas": p_valor < nivel_significancia
    }
    
    return resultado
