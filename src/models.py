import logging

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


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
