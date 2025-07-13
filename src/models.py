import logging

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit


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

def run_kfold_cv(model, X, y, k=5, scoring="neg_mean_squared_error", time_series=False):
    """
    Executa validação cruzada K-fold (ou TimeSeriesSplit se time_series=True) para um modelo.
    Retorna métricas por fold.
    """
    if time_series:
        cv = TimeSeriesSplit(n_splits=k)
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores