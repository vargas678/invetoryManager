# File: Modelo_.py
# Author: victo
# Copyright: 2024, Smart Cities Peru.
# License: MIT
#
# Purpose:
# This is the entry point for the application.
#
# Last Modified: 2024-05-03

# ================================== Importando Librerias =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pyod.models.knn import KNN

# ==========================================  Manejo de datos  =====================================================
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
        detector = KNN()
        data = df[[column]].values
        detector.fit(data)
        pred = detector.labels_
        outliers[column] = pred.sum()

    columnas_con_outliers = [col for col, num_outliers in outliers.items() if num_outliers > 0.05 * len(df)]
    return columnas_con_outliers


def handle_outliers(column, cap_value=None):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    if cap_value is not None:
        column = column.clip(lower=lower_limit, upper=cap_value)
    else:
        column = column.clip(lower=lower_limit, upper=upper_limit)

    return column


def detectar_columnas_ruidosas(df, threshold=0.05):
    columnas_ruidosas = []
    total_filas = len(df)

    for columna in df.columns:
        num_valores_unicos = df[columna].nunique()
        proporcion_valores_unicos = num_valores_unicos / total_filas

        if proporcion_valores_unicos < threshold:
            columnas_ruidosas.append(columna)

    return columnas_ruidosas


def flujo_completo(dataset_file, target_column):
    df = pd.read_excel(dataset_file)

    columnas_outliers_pyod = detectar_outliers_pyod(df)

    for columna in columnas_outliers_pyod:
        df[columna] = handle_outliers(df[columna])

    columnas_ruidosas = detectar_columnas_ruidosas(df)
    df = df.drop(columns=columnas_ruidosas)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = normalizar_estandarizar(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df


# ==========================================  Configurar la página =====================================================
st.set_page_config(
    page_title="Modelado",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

with open("./utils/Baner.html", "r", encoding="utf-8") as file:
    custom_banner_html = file.read()

with open("./utils/Baner_style.css", "r", encoding="utf-8") as file:
    custom_styles_css = file.read()

st.markdown("""
    <style>
        %s
    </style>
""" % custom_styles_css, unsafe_allow_html=True)

st.markdown(custom_banner_html, unsafe_allow_html=True)

st.markdown("## :date: Modelo RandomForest Regression")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# ====================================================== MANEJO DE DATOS ===============================================
dataset_file = "./data/Melsol-test.xlsx"
target_column = 'PRODUCTOS VENDIDOS'
X_train, X_test, y_train, y_test, df_sin_ruido = flujo_completo(dataset_file, target_column)

# ====================================================== ENTRENAMIENTO =================================================
bosque = RandomForestRegressor(
    n_estimators=100,
    criterion="squared_error",
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42
)

bosque.fit(X_train, y_train)

metrica = make_scorer(mean_squared_error, greater_is_better=False)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
resultados_cross_val = cross_val_score(bosque, X_train, y_train, cv=kf, scoring=metrica)

r2_full_set = bosque.score(X_train, y_train)
oob_score = bosque.oob_score_

y_pred = bosque.predict(X_test)
r2_test_set = r2_score(y_test, y_pred)
mae_value = mean_absolute_error(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)
rmse_value = np.sqrt(mse_value)
r2_value = r2_test_set

# ====================================================== GRAFICANDO ===================================================

# Gráfico de dispersión (Scatter Plot) con Plotly
scatter_fig = px.scatter(x=y_test, y=y_pred, title="Gráfico de Dispersión",
                         labels={"x": "Data Actual", "y": "Predicciones"})

# Curva de regresión
x_range = np.linspace(min(y_test), max(y_test), 100)
y_range = x_range  # Línea de regresión ideal (y = x)
scatter_fig.add_scatter(x=x_range, y=y_range, mode='lines', name='Línea de Regresión Ideal',
                        line=dict(color='red', dash='dash'))


def plot_metric(metrics_dict):
    fig = go.Figure()

    for label, value in metrics_dict.items():
        fig.add_trace(
            go.Indicator(
                value=value,
                gauge={"axis": {"visible": False}},
                number={
                    "prefix": "",
                    "suffix": "",
                    "font.size": 28,
                },
                title={
                    "text": label,
                    "font": {"size": 24},
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
        indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


# Crear dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: Gráfico de dispersión
with col1:
    col1.header("Gráfico de Dispersión")
    col1.plotly_chart(scatter_fig)

# Columna derecha: Métricas de rendimiento
with col2:
    col2.header("Métricas de Rendimiento")
    # Supongamos que ya calculaste tus métricas
    mae_value = mae_value
    mse_value = mse_value
    rmse_value = rmse_value
    r2_value = abs(r2_value)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # Llama a la función plot_gauge para cada métrica
    plot_gauge(mae_value, "blue", "", "Mean Absolute Error", 20)
    plot_gauge(mse_value, "green", "", "Mean Squared Error", 100)
    plot_gauge(rmse_value, "orange", "", "Root Mean Squared Error", 15)
    plot_gauge(r2_value, "red", "", "R²", 1)
# ============================================== CARACTERISTICAS DEL MODELO ==========================================
with col1:
    col1.header("Caracteristicas del Modelo")
    st.markdown("### METRICAS DEL MODELO")
    # Estilo personalizado con HTML
    style = """
        <style>
            .metrics-box {
                background-color: #363636;
                color: white;
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #2c2c2c;
                font-family: Arial, sans-serif;
            }
        </style>
    """

    # Construir el texto de las métricas
    metrics_text = f"<div class='metrics-box'>"
    metrics_text += f"<p><strong>Mean Absolute Error (MAE):</strong> {mae:.2f}</p>"
    metrics_text += f"<p><strong>Mean Squared Error (MSE):</strong> {mse:.2f}</p>"
    metrics_text += f"<p><strong>Root Mean Squared Error (RMSE):</strong> {rmse:.2f}</p>"
    metrics_text += f"<p><strong>R² (R cuadrado):</strong> {abs(r2):.2f}</p>"
    metrics_text += "</div>"

    # Agregar el estilo y las métricas al contenedor HTML
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(metrics_text, unsafe_allow_html=True)

    # Obtener características importantes del modelo (importances)
    importances = bosque.feature_importances_

    # Crear un DataFrame para mostrar características importantes
    st.markdown("### CARACTERISTICAS")
    importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    # Gráfico de barras para mostrar características importantes con Plotly Express
    bar_fig = px.bar(
        importances_df,
        x='Importance',
        y='Feature',
        orientation='h',  # Orientación horizontal
        title="Importancia de las Características",
        labels={'Importance': 'Importancia', 'Feature': 'Característica'},
        color='Importance',  # Colorea las barras según la importancia
        color_continuous_scale='Viridis',  # Puedes cambiar el esquema de color aquí
    )

    # Ajustes adicionales para mejorar la visualización
    bar_fig.update_layout(
        xaxis_title='Importancia',
        yaxis_title='Característica',
        showlegend=False,  # Oculta la leyenda, ya que el color indica la importancia
        margin=dict(l=10, r=10, b=10, t=30),  # Ajusta los márgenes para una mejor visualización
    )

    # Métricas de rendimiento
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics_text = (f"Mean Absolute Error: "
                    f"{mae:.2f}\nMean Squared Error: {mse:.2f}\nRoot Mean Squared Error: {rmse:.2f}\nR²: {r2:.2f}")

    # Impresión del modelo y resultados
    st.plotly_chart(bar_fig)
