# lab_functions.py
import numpy as np


def computeCost(X, y, theta):
    """
    Вычисляет функцию стоимости для линейной регрессии.
    """
    m = len(y)
    predictions = X.dot(theta)
    # Убедимся, что y - это вектор-столбец
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    sqrErrors = (predictions - y) ** 2
    J = (1 / (2 * m)) * np.sum(sqrErrors)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Выполняет градиентный спуск для поиска theta.
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    # Убедимся, что y - это вектор-столбец
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta = theta - (alpha * delta)
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


def featureNormalize(X):
    """
    Нормализует признаки (X).
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def normalEqn(X, y):
    """
    Вычисляет theta с помощью нормального уравнения.
    """
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta