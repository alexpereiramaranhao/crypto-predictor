"""
Funções para análise de lucro e métricas estatísticas.
"""

from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd


def calcular_correlacao(precos_reais: np.ndarray, previsoes: np.ndarray) -> float:
    """
    Calcula a correlação entre preços reais e previsões.
    
    Args:
        precos_reais: Array com preços reais
        previsoes: Array com previsões do modelo
    
    Returns:
        correlacao: Coeficiente de correlação
    """
    correlacao = np.corrcoef(precos_reais, previsoes)[0, 1]
    return correlacao


def calcular_erro_padrao(precos_reais: np.ndarray, previsoes: np.ndarray) -> float:
    """
    Calcula o erro padrão (desvio padrão dos erros).
    
    Args:
        precos_reais: Array com preços reais
        previsoes: Array com previsões do modelo
    
    Returns:
        erro_padrao: Erro padrão
    """
    erros = precos_reais - previsoes
    erro_padrao = np.std(erros)
    return erro_padrao


def mostrar_equacao_linear(modelo_linear) -> str:
    """
    Mostra a equação de um modelo linear.
    
    Args:
        modelo_linear: Modelo LinearRegression treinado
    
    Returns:
        equacao: String com a equação
    """
    intercepto = modelo_linear.intercept_
    coeficientes = modelo_linear.coef_
    
    # Monta equação
    equacao = f"y = {intercepto:.4f}"
    for i, coef in enumerate(coeficientes):
        equacao += f" + {coef:.4f}*x{i+1}"
    
    return equacao


def calcular_diferenca_erro_modelos(erro_modelo1: float, erro_modelo2: float) -> float:
    """
    Calcula a diferença absoluta entre erros de dois modelos.
    
    Args:
        erro_modelo1: Erro do primeiro modelo
        erro_modelo2: Erro do segundo modelo
    
    Returns:
        diferenca: Diferença absoluta entre os erros
    """
    diferenca = abs(erro_modelo1 - erro_modelo2)
    return diferenca


def imprimir_metricas_modelo(nome_modelo: str, precos_reais: np.ndarray, previsoes: np.ndarray) -> Dict[str, float]:
    """
    Calcula e imprime todas as métricas de um modelo.
    
    Args:
        nome_modelo: Nome do modelo
        precos_reais: Array com preços reais
        previsoes: Array com previsões
    
    Returns:
        metricas: Dicionário com todas as métricas
    """
    print(f"\n=== MÉTRICAS DO {nome_modelo.upper()} ===")
    
    # Calcula métricas
    correlacao = calcular_correlacao(precos_reais, previsoes)
    erro_padrao = calcular_erro_padrao(precos_reais, previsoes)
    
    print(f"Correlação: {correlacao:.4f}")
    print(f"Erro padrão: {erro_padrao:.4f}")
    
    metricas = {
        'correlacao': correlacao,
        'erro_padrao': erro_padrao
    }
    
    return metricas


def comparar_todos_modelos(resultados_modelos: Dict[str, Dict]) -> None:
    """
    Compara todos os modelos e mostra qual é o melhor.
    
    Args:
        resultados_modelos: Dicionário com resultados de cada modelo
    """
    print("\n" + "="*50)
    print("COMPARAÇÃO FINAL DOS MODELOS")
    print("="*50)
    
    # Encontra melhor modelo por correlação
    melhor_correlacao = 0
    modelo_melhor_correlacao = ""
    
    # Encontra melhor modelo por menor erro
    menor_erro = float('inf')
    modelo_menor_erro = ""
    
    for nome_modelo, metricas in resultados_modelos.items():
        correlacao = metricas['correlacao']
        erro = metricas['erro_padrao']
        
        print(f"{nome_modelo}: Correlação={correlacao:.4f}, Erro={erro:.4f}")
        
        if correlacao > melhor_correlacao:
            melhor_correlacao = correlacao
            modelo_melhor_correlacao = nome_modelo
            
        if erro < menor_erro:
            menor_erro = erro
            modelo_menor_erro = nome_modelo
    
    print(f"\nMelhor correlação: {modelo_melhor_correlacao} ({melhor_correlacao:.4f})")
    print(f"Menor erro: {modelo_menor_erro} ({menor_erro:.4f})")