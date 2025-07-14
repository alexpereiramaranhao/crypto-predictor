import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error


def train_mlp(X_train, y_train) -> MLPRegressor:
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


def train_linear(X_train, y_train) -> LinearRegression:
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


def validacao_cruzada_kfold(dados_X, dados_y, numero_folds=5):
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
        
        # Treina um modelo linear simples
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


def treinar_regressao_polinomial(dados_X, dados_y, grau_polinomio=2):
    """
    Treina um modelo de regressão polinomial bem simples.
    
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
