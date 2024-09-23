import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_sample(data, n=None):
    """
    Создает бутстрэп-выборку из данных.

    Args:
    data (pd.Series): Исходные данные для бутстрэппинга.
    n (int): Размер выборки. Если None, то используется размер исходных данных.

    Returns:
    np.ndarray: Бутстрэп-выборка.
    """
    if n is None:
        n = len(data)
    return np.random.choice(data, size=n, replace=True)


def bootstrap_t_test(data1, data2, n_iterations=1000):
    """
    Выполняет бутстрэппинг для t-теста.

    Args:
    data1 (pd.Series): Данные первой группы.
    data2 (pd.Series): Данные второй группы.
    n_iterations (int): Количество бутстрэп-выборок.

    Returns:
    float: p-значение для t-теста.
    """
    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    count = 0

    for _ in range(n_iterations):
        bootstrap1 = bootstrap_sample(combined, len(data1))
        bootstrap2 = bootstrap_sample(combined, len(data2))
        bootstrap_diff = np.mean(bootstrap1) - np.mean(bootstrap2)
        count += (bootstrap_diff >= observed_diff)

    p_value = count / n_iterations
    return observed_diff, p_value


def perform_statistical_analysis(data):
    """
    Выполняет бутстрэппинг для сравнения музыкальных параметров до и после 2000 года.

    Args:
    data (pd.DataFrame): отфильтрованные данные.

    Returns:
    None
    """
    # Разделение данных на две группы: до и после 2000 года
    data_90s = data[(data['year'] >= 1990) & (data['year'] < 2000)]
    data_00s = data[data['year'] >= 2000]

    features = ['tempo', 'duration_ms', 'instrumentalness', 'danceability', 'energy', 'acousticness', 'liveness']

    for feature in features:
        observed_diff, p_value = bootstrap_t_test(data_90s[feature], data_00s[feature])
        print(f"Бутстрэп: Разность для {feature}: {observed_diff}, p-значение: {p_value}")

