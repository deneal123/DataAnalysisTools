import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm


def analyze_mode_vs_valence(df):
    """
    Анализ корреляции между тональностью/модальностью (mode) и валентностью (valence).

    Args:
    df (pd.DataFrame): Датасет, содержащий столбцы mode и valence.

    Returns:
    None
    """
    # Корреляция между модальностью (mode) и валентностью (valence)
    mode_valence_corr, mode_valence_p = pearsonr(df['mode'], df['valence'])
    print(f"Корреляция между модальностью (mode) и валентностью (valence): {mode_valence_corr}, p-значение: {mode_valence_p}")

    # Визуализация корреляции через scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='mode', y='valence', data=df)
    plt.title('Mode vs Valence')
    plt.xlabel('Мажор/Минор (Mode)')
    plt.ylabel('Валентность (Valence)')
    plt.xticks([0, 1], ['Минор', 'Мажор'])
    plt.show()

    # Построение регрессионной модели для анализа влияния mode на valence
    X = df[['mode']]
    X = sm.add_constant(X)  # Добавляем константу для модели
    y = df['valence']

    model = sm.OLS(y, X).fit()
    print(model.summary())
