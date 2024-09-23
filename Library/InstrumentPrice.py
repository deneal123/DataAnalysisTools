import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_catboost_model(df, target_column):
    # Заменяем None на 0
    df.fillna(0, inplace=True)

    # Отделяем целевую переменную и признаки
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Определяем категориальные признаки
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Разделяем данные на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, cat_features=categorical_cols, verbose=0)
    model.fit(X_train, y_train)

    # Прогнозируем и оцениваем модель
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}, R²: {r2}')

    # Получаем важность признаков
    feature_importances = model.get_feature_importance()
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df
