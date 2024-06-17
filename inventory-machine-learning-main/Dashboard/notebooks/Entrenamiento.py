import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from pyod.models.knn import KNN


def normalizar_estandarizar(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    standardizer = StandardScaler()
    df_standardized = pd.DataFrame(standardizer.fit_transform(df_normalized), columns=df.columns)

    return df_standardized


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    mse = mean_squared_error(y_test, predicciones)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicciones)
    return mae, mse, rmse, r2


def detectar_outliers_pyod(df):
    outliers = {}
    for column in df.columns:
        # Utilizar KNN para la detección de outliers
        detector = KNN()
        data = df[[column]].values
        detector.fit(data)
        # Predicción: 0 significa normal, 1 significa outlier
        pred = detector.labels_
        outliers[column] = pred.sum()  # Número de outliers en la columna

    # Filtrar columnas que tienen más del 5% de datos como outliers
    columnas_con_outliers = [col for col, num_outliers in outliers.items() if num_outliers > 0.05 * len(df)]

    return columnas_con_outliers


def handle_outliers(column, cap_value=None):
    # Calcula el rango intercuartílico (IQR)
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    # Define los límites superior e inferior para identificar los valores atípicos
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Acotar los valores atípicos
    if cap_value is not None:
        column = column.clip(lower=lower_limit, upper=cap_value)
    else:
        column = column.clip(lower=lower_limit, upper=upper_limit)

    return column


def detectar_columnas_ruidosas(df, threshold=0.05):
    """
    Esta función identifica las columnas ruidosas en un DataFrame basándose en la proporción de valores únicos.

    Args:
    - df: DataFrame de pandas.
    - threshold: Umbral para considerar una columna como ruidosa. El valor predeterminado es 0.05 (5%).

    Returns:
    - columnas_ruidosas: Lista de nombres de columnas consideradas ruidosas.
    """
    columnas_ruidosas = []
    total_filas = len(df)

    for columna in df.columns:
        num_valores_unicos = df[columna].nunique()
        proporcion_valores_unicos = num_valores_unicos / total_filas

        if proporcion_valores_unicos < threshold:
            columnas_ruidosas.append(columna)

    return columnas_ruidosas


def flujo_completo(dataset_file, target_column):
    # Leer el archivo Excel
    df = pd.read_excel(dataset_file)

    # Detectar columnas con outliers usando PyOD
    columnas_outliers_pyod = detectar_outliers_pyod(df)

    # Iterar sobre las columnas con outliers y manejarlos
    for columna in columnas_outliers_pyod:
        # Manejar outliers en la columna actual
        df[columna] = handle_outliers(df[columna])

    # Detectar columnas ruidosas usando pandas-profiling
    columnas_ruidosas = detectar_columnas_ruidosas(df)

    # Eliminar las columnas ruidosas del DataFrame
    df = df.drop(columns=columnas_ruidosas)

    # Separar las características y el objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalizar y estandarizar los datos
    X = normalizar_estandarizar(X)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos a evaluar
    modelos = {
        'Regresión Lineal': LinearRegression(),
        'Lasso Regression': Lasso(),
        'Support Vector Regression': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVM': SVR(),
        'KNN': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(random_state=42),
        'Ridge Linear Regression': Ridge(),
        'Elastic-Net Regression': ElasticNet(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    }

    resultados = []

    for nombre, modelo in modelos.items():
        mae, mse, rmse, r2 = evaluar_modelo(modelo, X_train, y_train, X_test, y_test)
        resultados.append([nombre, mae, mse, rmse, r2])

    # Evaluación especial para Random Forest con validación cruzada
    bosque = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_features="sqrt", bootstrap=True,
                                   oob_score=True, random_state=42)
    metrica = make_scorer(mean_squared_error, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados_cross_val = cross_val_score(bosque, X, y, cv=kf, scoring=metrica)

    bosque.fit(X, y)
    predicciones_rf = bosque.predict(X_test)
    mae_rf = mean_absolute_error(y_test, predicciones_rf)
    mse_rf = mean_squared_error(y_test, predicciones_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, predicciones_rf)

    resultados.append(['Random Forest Regression', mae_rf, mse_rf, rmse_rf, r2_rf])

    # Crear un DataFrame con las métricas de los modelos
    resultados_df = pd.DataFrame(resultados, columns=['Modelo', 'MAE', 'MSE', 'RMSE', 'R²'])

    # Ordenar los resultados por R²
    resultados_df = resultados_df.sort_values(by='R²', ascending=False)

    return resultados_df


# Ejemplo de uso
dataset_file = "../Data/Melsol-test.xlsx"
target_column = 'PRODUCTOS VENDIDOS'
resultados = flujo_completo(dataset_file, target_column)
print(resultados)
