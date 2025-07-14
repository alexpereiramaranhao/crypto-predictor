import logging
from typing import Tuple, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error


def train_mlp(X_train: np.ndarray, y_train: np.ndarray) -> MLPRegressor:
    """
    Treina um modelo MLPRegressor.
    Returns:
        MLPRegressor: Modelo treinado.
    """
    try:
        model = MLPRegressor(random_state=42, max_iter=500)
        model.fit(X_train, y_train)
        logging.info("MLPRegressor treinado com sucesso!")
        return model
    except Exception as e:
        logging.error(f"Erro ao treinar o MLPRegressor: {e}")
        raise


def train_linear(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Treina um modelo LinearRegression.
    Returns:
        LinearRegression: Modelo treinado.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("LinearRegression treinado com sucesso!")
        return model
    except Exception as e:
        logging.error(f"Erro ao treinar o LinearRegression: {e}")
        raise


def validacao_cruzada_kfold(dados_X: np.ndarray, dados_y: np.ndarray, numero_folds: int = 5) -> Tuple[List[float], float]:
    """
    Função para fazer validação cruzada K-fold.
    
    Args:
        dados_X: Os dados de entrada (features)
        dados_y: Os dados de saída (target)
        numero_folds: Quantas partes dividir os dados (padrão: 5)
    
    Returns:
        lista_erros: Lista com os erros de cada fold
        erro_medio: Média dos erros
    """
    # Cria o objeto para dividir os dados em folds
    kfold = KFold(n_splits=numero_folds, shuffle=True, random_state=42)
    
    # Lista para guardar os erros de cada fold
    lista_erros = []
    
    print(f"Fazendo validação cruzada com {numero_folds} folds...")
    
    # Para cada divisão dos dados
    for numero_fold, (indices_treino, indices_teste) in enumerate(kfold.split(dados_X)):
        # Separa dados de treino e teste
        X_treino = dados_X[indices_treino]
        X_teste = dados_X[indices_teste]
        y_treino = dados_y[indices_treino]
        y_teste = dados_y[indices_teste]
        
        # Escala as features para evitar overflow
        scaler = StandardScaler()
        X_treino_escalado = scaler.fit_transform(X_treino)
        X_teste_escalado = scaler.transform(X_teste)
        
        # Treina um modelo linear
        modelo_linear = LinearRegression()
        modelo_linear.fit(X_treino_escalado, y_treino)
        
        # Faz previsões no conjunto de teste
        previsoes = modelo_linear.predict(X_teste_escalado)
        
        # Calcula o erro (RMSE - Root Mean Square Error)
        erro_rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
        lista_erros.append(erro_rmse)
        
        print(f"Fold {numero_fold + 1}: Erro RMSE = {erro_rmse:.4f}")
    
    # Calcula a média dos erros
    erro_medio = np.mean(lista_erros)
    desvio_padrao_erro = np.std(lista_erros)
    
    print(f"Erro médio da validação cruzada: {erro_medio:.4f} ± {desvio_padrao_erro:.4f}")
    
    return lista_erros, erro_medio


def treinar_regressao_polinomial(dados_X: np.ndarray, dados_y: np.ndarray, grau_polinomio: int = 2) -> Tuple[LinearRegression, PolynomialFeatures]:
    """
    Treina um modelo de regressão polinomial.
    
    Args:
        dados_X: Dados de entrada
        dados_y: Dados de saída
        grau_polinomio: Grau do polinômio (2, 3, ou 4)
    
    Returns:
        modelo_treinado: Modelo polinomial treinado
        transformador_polinomial: Para transformar novos dados
    """
    print(f"Treinando regressão polinomial de grau {grau_polinomio}...")
    
    # Cria as features polinomiais
    transformador_polinomial = PolynomialFeatures(degree=grau_polinomio)
    dados_X_polinomial = transformador_polinomial.fit_transform(dados_X)
    
    # Treina modelo linear com as features polinomiais
    modelo_linear = LinearRegression()
    modelo_linear.fit(dados_X_polinomial, dados_y)
    
    print(f"Modelo polinomial grau {grau_polinomio} treinado com sucesso!")
    
    return modelo_linear, transformador_polinomial


def encontrar_melhor_grau_polinomial(dados_X: np.ndarray, dados_y: np.ndarray) -> Tuple[int, float, LinearRegression, PolynomialFeatures]:
    """
    Testa graus polinomiais de 2 a 10 e encontra o melhor baseado no erro.
    
    Args:
        dados_X: Dados de entrada
        dados_y: Dados de saída
    
    Returns:
        melhor_grau: O grau que teve menor erro
        menor_erro: O menor erro encontrado
        melhor_modelo: O modelo treinado com melhor grau
        melhor_transformador: O transformador do melhor grau
    """
    print("Testando graus polinomiais de 2 a 10...")
    
    melhor_grau = 2
    menor_erro = float('inf')
    melhor_modelo = None
    melhor_transformador = None
    
    # Testa cada grau de 2 a 10
    for grau in range(2, 11):
        try:
            # Treina modelo polinomial com este grau
            modelo, transformador = treinar_regressao_polinomial(dados_X, dados_y, grau)
            
            # Faz validação cruzada para calcular erro
            _, erro_medio = validacao_cruzada_kfold(dados_X, dados_y, numero_folds=3)
            
            print(f"Grau {grau}: Erro médio = {erro_medio:.4f}")
            
            # Se este grau é melhor, guarda
            if erro_medio < menor_erro:
                menor_erro = erro_medio
                melhor_grau = grau
                melhor_modelo = modelo
                melhor_transformador = transformador
                
        except Exception as e:
            print(f"Erro no grau {grau}: {e}")
            continue
    
    print(f"Melhor grau polinomial: {melhor_grau} (erro: {menor_erro:.4f})")
    
    return melhor_grau, menor_erro, melhor_modelo, melhor_transformador
