# File: EDA_.py
# Author: victo
# Copyright: 2024, Smart Cities Peru.
# License: MIT
#
# Purpose:
# This is the entry point for the application.
#
# Last Modified: 2024-05-03

# ==========================================  Importaciones de bibliotecas ==================================

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.metric_cards import style_metric_cards
import altair as alt

# ==========================================  Configurar la página ==================================================
st.set_page_config(
    page_title="Análisis Exploratorio de Datos (EDA)",
    page_icon=":bar_chart:",
    layout="wide"
)
# ==================================================================================================================
# Cargar el componente de BannerPersonalizado.html
with open("./utils/Baner.html", "r", encoding="utf-8") as file:
    custom_banner_html = file.read()

# Agregar estilos CSS desde la carpeta utils
with open("./utils/Baner_style.css", "r", encoding="utf-8") as file:
    custom_styles_css = file.read()
# Mostrar el componente de Banner en Streamlit con los estilos CSS
st.markdown("""
    <style>
        %s
    </style>
""" % custom_styles_css, unsafe_allow_html=True)
st.markdown(custom_banner_html, unsafe_allow_html=True)

# ==========================================  Titulo de la Pagina ======================================================
st.markdown("## :bar_chart: Análisis Exploratorio de Datos (EDA) 👋")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# ================================================  Cargar Datos  ======================================================
# Leer el conjunto de datos por defecto
df = pd.read_excel("data/Melsol-test.xlsx", engine="openpyxl")

#  =========================================== Sidebar para realzar de los Filtros  =========================
# Definir nombres de los meses en español
meses_espanol = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

#  =========================================== Sidebar para realzar de los Filtros  =========================
st.sidebar.header("Configuración de Filtros:")

#  =========================================== POR MES  =========================
st.sidebar.subheader("Filtrar por Mes")
# Widget para seleccionar el mes
meses_seleccionados = st.sidebar.multiselect("Seleccionar Meses",
                                             list(meses_espanol.values()),
                                             default=list(meses_espanol.values()))
# Convertir los nombres de los meses seleccionados a números
meses = [list(meses_espanol.keys())[list(meses_espanol.values()).index(mes)] for mes in meses_seleccionados]

# Filtrar el DataFrame por los meses seleccionados
if meses:
    df = df[df["MES"].isin(meses)]

#  =========================================== POR RANGO DE PRODUCTOS ALMACENADOS =========================
st.sidebar.subheader("Filtrar por rango de Productos Almacenados")
# Obtener el valor máximo de la columna "PRODUCTOS ALMACENADOS"
max_productos_almacenados = df["PRODUCTOS ALMACENADOS"].max()
# Widget para filtrar por rango de Productos Almacenados
almacenados_range = st.sidebar.slider("Filtrar por Rango de Productos Almacenados",
                                      min_value=0,
                                      max_value=max_productos_almacenados,
                                      value=(0, max_productos_almacenados))

# Filtrar los datos por el rango seleccionado
df = df[(df["PRODUCTOS ALMACENADOS"] >= almacenados_range[0]) &
        (df["PRODUCTOS ALMACENADOS"] <= almacenados_range[1])]

#  =========================================== POR DEMANDA DEL PRODUCTO =========================
st.sidebar.subheader("Filtrar por Demanda del Producto")
demanda_options = df["DEMANDA DEL PRODUCTO"].unique()
selected_demanda = st.sidebar.multiselect("Filtrar por Demanda del Producto", demanda_options, default=demanda_options)
df = df[df["DEMANDA DEL PRODUCTO"].isin(selected_demanda)]

#  =========================================== POR FESTIVIDAD =========================
st.sidebar.subheader("Filtrar por Festividad")
# Definir el mapeo de valores de festividad a etiquetas deseadas
mapeo_festividad = {0: "NO HAY FESTIVIDAD", 1: "SI HAY FESTIVIDAD"}
# Widget para filtrar por festividad
festividad_seleccionada = st.sidebar.multiselect("Filtrar por Festividad",
                                                 options=list(mapeo_festividad.values()),
                                                 default=list(mapeo_festividad.values()))

# Convertir las etiquetas seleccionadas de festividad a los valores correspondientes
festividad_seleccionada = [list(mapeo_festividad.keys())[list(mapeo_festividad.values()).index(label)]
                           for label in festividad_seleccionada]

# Filtrar los datos por el rango seleccionado y la festividad seleccionada
df = df[(df["FESTIVIDAD"].isin(festividad_seleccionada))]

#  =============================================== Mostrar los datos filtrados =========================
with st.expander("Mostrar Conjunto de Datos"):
    df = dataframe_explorer(df, case=False)
    st.dataframe(df, use_container_width=True)

# =============================================== Mostrar estadísticas descriptivas =========================
st.subheader('Estadísticas Descriptivas:', divider='rainbow', )

st.write(df.describe())

#  =============================================== Mostrar los datos filtrados =========================
# Dividir el espacio horizontalmente en dos columnas
col1, col2 = st.columns(2)

# Gráfico de dispersión en la primera columna
with col1:
    st.subheader('Gráfico de Dispersión:', divider='rainbow', )
    scatter_fig = px.scatter(df, x="PRODUCTOS ALMACENADOS", y="PRODUCTOS VENDIDOS",
                             color="DEMANDA DEL PRODUCTO")
    st.plotly_chart(scatter_fig)

# Tarjetas métricas en la segunda columna
with col2:
    st.subheader('Métricas del Conjunto de Datos:', divider='rainbow', )
    col1, col2 = st.columns(2)

    # Calcular el total de productos almacenados
    total_productos_almacenados = df["PRODUCTOS ALMACENADOS"].sum()
    # Calcular el delta para el total de productos almacenados
    delta_productos_almacenados = total_productos_almacenados - df["PRODUCTOS VENDIDOS"].sum()
    # Convertir el delta a un tipo de dato compatible
    delta_productos_almacenados_str = str(delta_productos_almacenados)

    # Mostrar la métrica para el total de productos almacenados
    col1.metric(label="Total de Productos", value=total_productos_almacenados,
                delta="productos en stock: " + delta_productos_almacenados_str)

    # Calcular el promedio de precio de venta
    promedio_precio_venta = round(df["PRECIO DE VENTA"].mean(), 2)

    # Calcular el promedio total de gastos (almacenamiento y marketing)
    promedio_total_gastos = (df["GASTO DE MARKETING"].sum() + df["GASTO DE ALMACENAMIENTO"].sum()) / len(df)

    # Calcular el delta para el promedio de precio de venta con respecto al promedio total de gastos
    delta_promedio_precio_venta = round(promedio_precio_venta - promedio_total_gastos, 2)

    # Convertir el delta a un tipo de dato compatible
    delta_promedio_precio_venta_str = str(delta_promedio_precio_venta)

    # Mostrar la métrica para el promedio de precio de venta
    col1.metric(label="Promedio de Precio de Venta", value=promedio_precio_venta, delta="promedio ganancia: " +
                                                                                        delta_promedio_precio_venta_str)

    # Calcular la cantidad total de productos vendidos
    cantidad_total_productos_vendidos = round(df["DEMANDA DEL PRODUCTO"].mean(), 2)
    # Calcular el delta para la cantidad total de productos vendidos
    delta_cantidad_total_productos_vendidos = df["FESTIVIDAD"].sum()
    # Convertir el delta a un tipo de dato compatible
    delta_cantidad_total_productos_vendidos_str = str(delta_cantidad_total_productos_vendidos)

    # Mostrar la métrica para la cantidad total de productos vendidos
    col2.metric(label="Demanda del Producto", value=cantidad_total_productos_vendidos,
                delta="N° Festividades: " + delta_cantidad_total_productos_vendidos_str)

    # Calcular la tasa de crecimiento mensual
    df["TASA_DE_CRECIMIENTO"] = (
            df["PRODUCTOS ALMACENADOS"] * df["PRECIO DE VENTA"] * df["DEMANDA DEL PRODUCTO"]).pct_change()

    # Calcular el promedio de la tasa de crecimiento mensual
    promedio_tasa_crecimiento_mensual = round(df["TASA_DE_CRECIMIENTO"].mean(), 2)

    ultimo_registro = df.iloc[-1]
    # Calcular las ventas en el último mes
    ventas_en_ultimo_mes = round(ultimo_registro["DEMANDA DEL PRODUCTO"] * ultimo_registro["PRECIO DE VENTA"], 2)

    # Convertir el promedio de la tasa de crecimiento mensual a un tipo de dato compatible
    promedio_tasa_crecimiento_mensual_str = str(ventas_en_ultimo_mes)

    # Mostrar la métrica para el promedio de la tasa de crecimiento mensual
    col2.metric(label="Tasa de Crecimiento Mensual", value=promedio_tasa_crecimiento_mensual,
                delta= promedio_tasa_crecimiento_mensual_str + " ventas en el ultimo mes")

    # Calcular las ventas totales para cada mes multiplicando la demanda del producto por el precio de venta
    ventas_por_mes = df.groupby("MES").apply(lambda x: (x["DEMANDA DEL PRODUCTO"] * x["PRECIO DE VENTA"]).sum())
    # Encontrar el mes con mayores ventas
    mes_con_mas_ventas = ventas_por_mes.idxmax()
    # Calcular las ventas en el mes con más ventas
    ventas_en_mes_con_mas_ventas = round(ventas_por_mes.max(), 2)
    # Calcular el delta como la diferencia entre las ventas en el último mes y el mes con más ventas
    delta_ventas = round(ventas_en_ultimo_mes - ventas_en_mes_con_mas_ventas, 2)
    # Convertir el delta a un tipo de dato compatible
    delta_ventas_str = str(delta_ventas)

    # Mostrar la métrica para el mes con más ventas
    col1.metric(label="Mes con Mayor Ventas", value=f"{mes_con_mas_ventas} "
                                                    f"({ventas_en_mes_con_mas_ventas} $)",
                delta=delta_ventas_str + " tasa de ventas actual")

    # Calcular la ganancia total
    ingresos_total = round((df["DEMANDA DEL PRODUCTO"] * df["PRECIO DE VENTA"]).sum() - (
            df["GASTO DE MARKETING"].sum() + df["GASTO DE ALMACENAMIENTO"].sum()), 2)
    # Calcular el delta para la ganancia total
    delta_ganancia_total = round(ingresos_total - df["GASTO DE MARKETING"].sum(), 2)
    # Convertir el delta a un tipo de dato compatible
    delta_ganancia_total_str = str(delta_ganancia_total)

    # Mostrar la métrica para la ganancia total
    col2.metric(label="Ingreso Total", value=ingresos_total, delta="ganancia: " + delta_ganancia_total_str)

    # Estilo de las tarjetas métricas
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    from streamlit_extras.metric_cards import style_metric_cards

    style_metric_cards(background_color="#596073", border_left_color="#F71938", border_color="#1f66bd",
                       box_shadow="#F71938")


# Subdividir el espacio disponible en dos columnas
a1, a2 = st.columns(2)

# Dentro de la columna a1
with a1:
    # Título para el gráfico interactivo de círculos
    st.subheader('Gasto de Almacenamiento vs Productos Almacenados', divider='rainbow')

    # Crear un gráfico interactivo de círculos con Altair
    scatter_chart = alt.Chart(df).mark_circle().encode(
        x='PRODUCTOS ALMACENADOS',
        y='GASTO DE ALMACENAMIENTO',
        color='MES:N'
    ).interactive()

    # Mostrar el gráfico interactivo de círculos
    st.altair_chart(scatter_chart, use_container_width=True)

# Dentro de la columna a2
with a2:
    # Título para el gráfico de barras
    st.subheader('Productos Vendidos por Mes', divider='rainbow')

    # Crear un gráfico de barras con Altair
    bar_chart = alt.Chart(df).mark_bar().encode(
        x='MES:N',
        y='PRODUCTOS VENDIDOS',
        color='FESTIVIDAD:Q'
    )

    # Mostrar el gráfico de barras
    st.altair_chart(bar_chart, use_container_width=True)