import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm


def bootstrap_test(data, n_iterations=1000):
    """
    Функция для выполнения бутстреп-теста и вычисления доверительного интервала.

    Args:
    data (array-like): Данные для бутстрепа.
    n_iterations (int): Количество итераций бутстрепа.

    Returns:
    mean (float): Среднее значение.
    lower_ci (float): Нижний предел доверительного интервала.
    upper_ci (float): Верхний предел доверительного интервала.
    """
    means = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    mean = np.mean(means)
    lower_ci = np.percentile(means, 2.5)
    upper_ci = np.percentile(means, 97.5)

    return mean, lower_ci, upper_ci


def analyze_key_diversity(df):
    """
    Анализ разнообразия тональностей (key) за последнее время с использованием бутстрепа.

    Args:
    df (pd.DataFrame): Датасет, содержащий столбцы year и key.

    Returns:
    None
    """
    # Группировка по годам и подсчет уникальных ключей
    key_diversity = df.groupby('year')['key'].nunique().reset_index()
    key_diversity.columns = ['year', 'unique_keys']

    # Бутстреп-тест для уникальных ключей
    unique_keys_mean, unique_keys_lower_ci, unique_keys_upper_ci = bootstrap_test(key_diversity['unique_keys'].values)
    print(
        f"Уникальные ключи: Среднее = {unique_keys_mean}, Доверительный интервал = [{unique_keys_lower_ci}, {unique_keys_upper_ci}]")

    # Расчет Симпсонова индекса
    diversity_index = df.groupby('year')['key'].value_counts(normalize=True).reset_index(name='proportion')
    diversity_index['diversity'] = diversity_index.groupby('year')['proportion'].transform(lambda x: 1 - sum(x ** 2))
    average_diversity = diversity_index.groupby('year')['diversity'].mean().reset_index()

    # Бутстреп-тест для Симпсонова индекса
    simpson_index_mean, simpson_index_lower_ci, simpson_index_upper_ci = bootstrap_test(
        average_diversity['diversity'].values)
    print(
        f"Симпсонов индекс: Среднее = {simpson_index_mean}, Доверительный интервал = [{simpson_index_lower_ci}, {simpson_index_upper_ci}]")

    # Построение графиков аналогично предыдущему коду
    plt.figure(figsize=(12, 6))
    plt.plot(key_diversity['year'], key_diversity['unique_keys'], marker='o')
    plt.title('Разнообразие ключей тональностей по годам')
    plt.xlabel('Год')
    plt.ylabel('Количество уникальных ключей')
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(average_diversity['year'], average_diversity['diversity'], marker='o', color='orange')
    plt.title('Средний Симпсонов индекс разнообразия ключей по годам')
    plt.xlabel('Год')
    plt.ylabel('Симпсонов индекс разнообразия')
    plt.grid()
    plt.show()

