import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def analyze_joint_influence(df):
    """
    Анализ совместного влияния tempo, mode и loudness на valence с учетом стандартизации.

    Args:
    df (pd.DataFrame): Датасет, содержащий столбцы valence, tempo, mode, loudness.

    Returns:
    None
    """
    # Выбор нужных переменных
    X = df[['tempo', 'mode', 'loudness']]
    y = df['valence']

    # Стандартизация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Добавление константы для свободного члена
    X_scaled = sm.add_constant(X_scaled)

    # Построение модели OLS
    model = sm.OLS(y, X_scaled).fit()

    # Вывод результатов
    print(model.summary())

# Пример вызова функции
# analyze_joint_influence(df)
