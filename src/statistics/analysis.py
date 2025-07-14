import logging
from typing import Dict

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
        
        # TESTE POST-HOC (Tukey HSD)
        print("\n🔍 Realizando teste post-hoc de Tukey HSD...")
        
        # Prepara dados para Tukey HSD
        dados_tukey = []
        grupos_tukey = []
        
        for i, grupo in enumerate(grupos_retornos):
            dados_tukey.extend(grupo)
            grupos_tukey.extend([nomes_criptos[i]] * len(grupo))
        
        # Executa Tukey HSD
        resultado_tukey = pairwise_tukeyhsd(dados_tukey, grupos_tukey, alpha=nivel_significancia)
        
        print("Resultado do teste de Tukey HSD:")
        print(resultado_tukey)
        
        # Mostra resumo simples
        print(f"\nForam encontradas {sum(resultado_tukey.reject)} comparações significativas entre as criptomoedas.")
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


def anova_grupos_caracteristicas(dados_criptos_dict):
    """
    ANOVA para comparar retornos médios entre grupos de criptomoedas agrupadas por volatilidade.
    
    Agrupa as criptomoedas em 3 grupos (alta, média, baixa volatilidade) e compara 
    se há diferenças significativas nos retornos médios entre os grupos.
    
    Args:
        dados_criptos_dict: Dicionário com {nome_crypto: dataframe}
                           Cada dataframe deve ter colunas 'retorno_diario' e 'volatilidade_7d'
    
    Returns:
        resultado: Dicionário com os resultados da ANOVA entre grupos
    """
    print("Realizando ANOVA entre grupos de volatilidade...")
    
    # 1. Calcular volatilidade média de cada criptomoeda
    volatilidades_criptos = {}
    retornos_criptos = {}
    
    for nome_crypto, dataframe in dados_criptos_dict.items():
        if 'volatilidade_7d' in dataframe.columns and 'retorno_diario' in dataframe.columns:
            volatilidade_media = dataframe['volatilidade_7d'].dropna().mean()
            retornos_limpos = dataframe['retorno_diario'].dropna()
            
            if not np.isnan(volatilidade_media) and len(retornos_limpos) > 0:
                volatilidades_criptos[nome_crypto] = volatilidade_media
                retornos_criptos[nome_crypto] = retornos_limpos.tolist()
                print(f"{nome_crypto}: Volatilidade média = {volatilidade_media:.4f}%, Retorno médio = {retornos_limpos.mean():.4f}%")
    
    if len(volatilidades_criptos) < 3:
        print("ERRO: Precisa de pelo menos 3 criptomoedas para formar grupos!")
        return None
    
    # 2. Dividir em 3 grupos usando tercis (33% e 67%)
    volatilidades_valores = list(volatilidades_criptos.values())
    percentil_33 = np.percentile(volatilidades_valores, 33.33)
    percentil_67 = np.percentile(volatilidades_valores, 66.67)
    
    print(f"\nDivisão dos grupos por volatilidade:")
    print(f"Baixa volatilidade: ≤ {percentil_33:.4f}%")
    print(f"Média volatilidade: {percentil_33:.4f}% < vol ≤ {percentil_67:.4f}%")
    print(f"Alta volatilidade: > {percentil_67:.4f}%")
    
    # 3. Classificar cada criptomoeda em um grupo
    grupo_baixa_vol = []
    grupo_media_vol = []
    grupo_alta_vol = []
    
    retornos_baixa_vol = []
    retornos_media_vol = []
    retornos_alta_vol = []
    
    for nome_crypto, volatilidade in volatilidades_criptos.items():
        retornos = retornos_criptos[nome_crypto]
        
        if volatilidade <= percentil_33:
            grupo_baixa_vol.append(nome_crypto)
            retornos_baixa_vol.extend(retornos)
        elif volatilidade <= percentil_67:
            grupo_media_vol.append(nome_crypto)
            retornos_media_vol.extend(retornos)
        else:
            grupo_alta_vol.append(nome_crypto)
            retornos_alta_vol.extend(retornos)
    
    print(f"\nGrupo Baixa Volatilidade: {grupo_baixa_vol} ({len(retornos_baixa_vol)} observações)")
    print(f"Grupo Média Volatilidade: {grupo_media_vol} ({len(retornos_media_vol)} observações)")
    print(f"Grupo Alta Volatilidade: {grupo_alta_vol} ({len(retornos_alta_vol)} observações)")
    
    # 4. Verificar se todos os grupos têm dados
    grupos_retornos = []
    nomes_grupos = []
    
    if len(retornos_baixa_vol) > 0:
        grupos_retornos.append(retornos_baixa_vol)
        nomes_grupos.append("Baixa Volatilidade")
    if len(retornos_media_vol) > 0:
        grupos_retornos.append(retornos_media_vol)
        nomes_grupos.append("Média Volatilidade")
    if len(retornos_alta_vol) > 0:
        grupos_retornos.append(retornos_alta_vol)
        nomes_grupos.append("Alta Volatilidade")
    
    if len(grupos_retornos) < 2:
        print("ERRO: Precisa de pelo menos 2 grupos com dados para fazer ANOVA!")
        return None
    
    # 5. Realizar ANOVA
    estatistica_f, p_valor = stats.f_oneway(*grupos_retornos)
    
    # 6. Calcular médias de cada grupo
    medias_grupos = [np.mean(grupo) for grupo in grupos_retornos]
    
    print(f"\nRetornos médios por grupo:")
    for i, nome_grupo in enumerate(nomes_grupos):
        print(f"{nome_grupo}: {medias_grupos[i]:.4f}%")
    
    print(f"\nEstatística F: {estatistica_f:.4f}")
    print(f"P-valor: {p_valor:.6f}")
    
    nivel_significancia = 0.05
    if p_valor < nivel_significancia:
        print(f"RESULTADO: Rejeitamos H0 (p < {nivel_significancia})")
        print("CONCLUSÃO: Há diferenças significativas entre os retornos médios dos grupos de volatilidade")
        
        # Mostra qual grupo tem maior retorno
        indice_maior = np.argmax(medias_grupos)
        grupo_melhor = nomes_grupos[indice_maior]
        print(f"Grupo com maior retorno médio: {grupo_melhor} ({medias_grupos[indice_maior]:.4f}%)")
        
        # TESTE POST-HOC (Tukey HSD)
        print("\n🔍 Realizando teste post-hoc de Tukey HSD...")
        
        # Prepara dados para Tukey HSD
        dados_tukey = []
        grupos_tukey = []
        
        for i, grupo in enumerate(grupos_retornos):
            dados_tukey.extend(grupo)
            grupos_tukey.extend([nomes_grupos[i]] * len(grupo))
        
        # Executa Tukey HSD
        resultado_tukey = pairwise_tukeyhsd(dados_tukey, grupos_tukey, alpha=nivel_significancia)
        
        print("Resultado do teste de Tukey HSD:")
        print(resultado_tukey)
        
        # Mostra resumo simples
        print(f"\nForam encontradas {sum(resultado_tukey.reject)} comparações significativas entre os grupos.")
    else:
        print(f"RESULTADO: Não rejeitamos H0 (p >= {nivel_significancia})")
        print("CONCLUSÃO: Não há diferenças significativas entre os retornos médios dos grupos")
    
    resultado = {
        "estatistica_f": estatistica_f,
        "p_valor": p_valor,
        "grupos": nomes_grupos,
        "medias_grupos": medias_grupos,
        "diferencas_significativas": p_valor < nivel_significancia,
        "grupo_baixa_vol": grupo_baixa_vol,
        "grupo_media_vol": grupo_media_vol,
        "grupo_alta_vol": grupo_alta_vol
    }
    
    return resultado
