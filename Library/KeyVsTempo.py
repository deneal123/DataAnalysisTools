import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm


def analyze_tonality_vs_tempo(df):
    """
    Анализ корреляции между тональностью (key), тональностью/модальностью (mode) и темпом (tempo).

    Args:
    df (pd.DataFrame): Датасет, содержащий столбцы key, mode и tempo.

    Returns:
    None
    """
    # Корреляция между тональностью (key) и темпом (tempo)
    key_tempo_corr, key_tempo_p = pearsonr(df['key'], df['tempo'])
    print(f"Корреляция между тональностью (key) и темпом (tempo): {key_tempo_corr}, p-значение: {key_tempo_p}")

    # Корреляция между тональностью/модальностью (mode) и темпом (tempo)
    mode_tempo_corr, mode_tempo_p = pearsonr(df['mode'], df['tempo'])
    print(f"Корреляция между тональностью/модальностью (mode) и темпом (tempo): {mode_tempo_corr}, p-значение: {mode_tempo_p}")

    # Построение регрессионной модели для анализа влияния key и mode на tempo
    X = df[['key', 'mode']]
    X = sm.add_constant(X)  # Добавляем константу для модели
    y = df['tempo']

    model = sm.OLS(y, X).fit()
    print(model.summary())


