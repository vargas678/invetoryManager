# File: Predicciones.py
# Author: victo
# Copyright: 2024, Smart Cities Peru.
# License: MIT
#
# Purpose:
# This is the entry point for the application.
#
# Last Modified: 2024-05-03

# ================================= Importando Librerias =============================
from pages.Modelo_ import bosque, X_train
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
from fpdf import FPDF
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyod.models.knn import KNN

# ========== Configurar la página ================
st.set_page_config(
    page_title="Predicciones de Ventas",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"  # Puedes ajustar según tus preferencias
)

# ==========================================  Baner ====================================================================
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
st.markdown("## :chart_with_upwards_trend: Predicciones")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Componente para cargar un archivo Excel
uploaded_file = st.file_uploader(":file_folder: Cargar archivo Excel", type=["xlsx", "xls"])
# ================================================  Cargar Datos  ======================================================
# Leer el conjunto de datos por defecto
df = pd.read_excel("./data/Diclofenaco-prediccion.xlsx", engine="openpyxl")
# Carga el archivo y crea un DataFrame si se ha cargado un archivo
if uploaded_file is not None:
    filename = uploaded_file.name
    st.write(filename)
    try:
        new_df = pd.read_excel(uploaded_file, engine="openpyxl")
        new_df.columns = df.columns
        #  verifica si la cantidad de columnas no es la misma.
        if len(df.columns) != len(new_df.columns):
            st.error("Error: El archivo cargado no tiene la misma cantidad de columnas que el archivo por defecto.")
        #  verifica si los tipos de datos de las columnas no son iguales.
        else:
            # Guardar y actualizar el nuevo DataFrame
            new_df.to_excel("./data/Diclofenaco-prediccion.xlsx", index=False)
            df = new_df
            st.success("El archivo se ha cargado con éxito.")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
else:
    st.warning("Ningún archivo se ha cargado, se utilizará un archivo por defecto.")


# ============================================================================================================================
def normalizar_estandarizar(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    standardizer = StandardScaler()
    df_standardized = pd.DataFrame(standardizer.fit_transform(df_normalized), columns=df.columns)

    return df_standardized


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


# Filtrar las características necesarias utilizando las columnas del modelo
columnas_necesarias = X_train.columns
nuevos_datos = df[columnas_necesarias]

# Aplicar el preprocesamiento de outliers y normalización/estandarización
for columna in columnas_necesarias:
    nuevos_datos[columna] = handle_outliers(nuevos_datos[columna])
nuevos_datos = normalizar_estandarizar(nuevos_datos)
# Generar predicciones
predicciones = bosque.predict(nuevos_datos)
predicciones = np.round(predicciones).astype(int)
# Crear un nuevo DataFrame con las predicciones
df_predicciones = pd.DataFrame({"PRODUCTOS VENDIDOS": predicciones})

# Agregar la columna de predicciones al DataFrame original
df_con_predicciones = pd.concat([df, df_predicciones], axis=1)

# Crear una tabla con Plotly sin formato de coma decimal en la columna 'PRODUCTOS VENDIDOS'
fig_predicciones = go.Figure(data=[go.Table(
    header=dict(values=['PRODUCTOS QUE SE VENDERÁN' if col == 'PRODUCTOS VENDIDOS' else col for col in
                        df_con_predicciones.columns],
                fill_color='darkslategray', line_color='white', align='center'),
    cells=dict(values=[
        df_con_predicciones[col] if col != 'PRODUCTOS VENDIDOS' else df_con_predicciones[col].apply(
            lambda x: f'{int(x)}' if not pd.isna(x) else '')
        for col in df_con_predicciones.columns
    ],
        fill_color=['dimgray' if col != 'PRODUCTOS VENDIDOS' else 'steelblue' for col in df_con_predicciones.columns],
        line_color='white', align='center',
        format=[None if col != 'PRODUCTOS VENDIDOS' else None for col in df_con_predicciones.columns])
)])

# Estilo adicional para la tabla
fig_predicciones.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
)

# Usar contenedores y columnas para organizar el diseño
container1 = st.container()
container2 = st.container()

with container1:
    # Dividir la página en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        # Mostrar la tabla con predicciones debajo de los gráficos
        st.subheader("Cantidad de Productos que se Venderán:")
        st.plotly_chart(fig_predicciones)
    with col2:
        # Crear gráfico de dispersión entre Demanda del Producto y Productos Vendidos Predichos
        scatter_predicciones = px.scatter(df_con_predicciones, x='DEMANDA DEL PRODUCTO', y='PRODUCTOS VENDIDOS',
                                          title='Diagrama de Dispersión: Demanda del Producto vs Productos Vendidos',
                                          labels={'DEMANDA DEL PRODUCTO': 'DEMANDA DEL PRODUCTO',
                                                  'PRODUCTOS VENDIDOS': 'PRODUCTOS VENDIDOS'})
        # Mostrar el gráfico de dispersión
        st.plotly_chart(scatter_predicciones)

with container2:
    # ------------------------ Realizar Predicciones ---------------------------
    # Formulario para ingresar nuevos datos y botón de predicción
    st.header("Ingresar Datos para Predicción")
    # Mostrar todas las variables en el formulario
    new_data = {
        "MES": st.number_input("MES", min_value=1, max_value=12, key="mes_input"),
        "PRODUCTOS ALMACENADOS": st.number_input("NUMERO DE PRODUCTOS ALMACENADOS", min_value=0),
        "GASTO DE MARKETING": st.number_input("GASTO DE MARKETING", min_value=0.0, step=1.0),
        "GASTO DE ALMACENAMIENTO": st.number_input("GASTO DE ALMACENAMIENTO", min_value=0.0, step=1.0),
        "DEMANDA DEL PRODUCTO": st.number_input("DEMANDA DEL PRODUCTO", min_value=1, max_value=10),
        "FESTIVIDAD": st.number_input("FESTIVIDAD", min_value=0, max_value=1),
        "PRECIO DE VENTA": st.number_input("PRECIO DE VENTA", min_value=0.0, step=1.0)
    }

    # Botón para realizar predicción
    if st.button("Realizar Predicción"):
        # Filtrar las variables que no son consideradas ruido
        new_data_filtered = {key: value for key, value in new_data.items() if key in X_train.columns}
        new_data_df = pd.DataFrame(new_data_filtered, index=[0])
        new_predictions = bosque.predict(new_data_df) + 10
        new_predictions = np.round(new_predictions).astype(int)
        # Estilo del cuadro de texto
        style = """
            padding: 10px;
            background-color: #001F3F; /* Color plateado o gris claro */
            border: 2px solid #C0C0C0; /* Borde del cuadro */
            border-radius: 5px; /* Esquinas redondeadas */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra */
            color: #001F3F; /* Texto en azul oscuro */
            font-size: 18px; /* Tamaño de fuente más grande */
        """

        # Imprimir predicción en el cuadro de texto con estilo
        rounded_prediction = int(round(new_predictions[0]))
        st.markdown(f'<div style="{style}">Predicción de Productos que se venderán: {rounded_prediction}</div>',
                    unsafe_allow_html=True)


# ------------------------ Preparation del Reporte en PDF ---------------------------
# Función para establecer el color de fondo en función del valor de la columna
def color_based_on_percentile(value, mean_value, p25, p50):
    if value > mean_value:
        color = 'background-color: #11F248; color: black'  # Verde oscuro para valores altos (por encima del promedio)
    elif value > p50:
        color = 'background-color: #8AF211; color: black'  # Verde claro para valores medianos (entre p50 y mean)
    elif value > p25:
        color = 'background-color: #C4F423; color: black'  # Amarillo para valores bajos (entre p25 y p50)
    else:
        color = 'background-color: #F4F141; color: black'  # Amarillo claro para valores muy bajos (menores al p25)
    return color


# Renombrar la columna 'PRODUCTOS VENDIDOS' a 'PRODUCTOS QUE SE VENDERÁN'
df_con_predicciones = df_con_predicciones.rename(columns={'PRODUCTOS VENDIDOS': 'PRODUCTOS QUE SE VENDERÁN'})


# Calcular promedio y percentiles para ambas columnas
def calculate_statistics(df, column):
    mean_value = df[column].mean()
    p25 = np.percentile(df[column], 25)
    p50 = np.percentile(df[column], 50)
    return mean_value, p25, p50


mean_value_almacenados, p25_almacenados, p50_almacenados = calculate_statistics(df_con_predicciones,
                                                                                'PRODUCTOS ALMACENADOS')
mean_value_vendidos, p25_vendidos, p50_vendidos = calculate_statistics(df_con_predicciones,
                                                                       'PRODUCTOS QUE SE VENDERÁN')

# Eliminar la columna del índice del DataFrame original
df_con_predicciones = df_con_predicciones.reset_index(drop=True)

# Extraer los datos del DataFrame
data = df_con_predicciones.values
columns = df_con_predicciones.columns

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(16, 6))  # Ajusta el tamaño de la figura según sea necesario
ax.axis('off')  # Oculta los ejes

# Convierte el DataFrame a una tabla de matplotlib
mpl_table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

# Aplicar estilo visualmente después
for (i, j), cell in mpl_table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', fontsize=12)  # Encabezados en negrita y tamaño de fuente reducido a 12
        cell.set_height(0.15)  # Aumentar altura de la celda del encabezado
        cell.set_fontsize(12)  # Tamaño de fuente para los encabezados
    else:
        value = df_con_predicciones.iat[i - 1, j]
        if j == df_con_predicciones.columns.get_loc('PRODUCTOS ALMACENADOS'):  # Columna 'PRODUCTOS ALMACENADOS'
            # Aplicar color basado en la función
            style = color_based_on_percentile(value, mean_value_almacenados, p25_almacenados, p50_almacenados)
            bg_color = style.split(';')[0].split(':')[1].strip()
            text_color = style.split(';')[1].split(':')[1].strip()
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=text_color)
        elif j == df_con_predicciones.columns.get_loc(
                'PRODUCTOS QUE SE VENDERÁN'):  # Columna 'PRODUCTOS QUE SE VENDERÁN'
            style = color_based_on_percentile(value, mean_value_vendidos, p25_vendidos, p50_vendidos)
            bg_color = style.split(';')[0].split(':')[1].strip()
            text_color = style.split(';')[1].split(':')[1].strip()
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=text_color)

# Aplicar bordes
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(10)  # Tamaño de fuente para las celdas
mpl_table.scale(1.5, 1.5)  # Escalar la tabla horizontalmente y verticalmente

# Guardar la imagen
plt.savefig('images/ventas_mensuales.png', bbox_inches='tight', pad_inches=0.1)
plt.close(fig)


def generate_matplotlib_line_chart(df, filename):
    # Define paleta de colores para festividad
    festividad_palette = {0: '#EB2D53', 1: '#2B21EB'}
    # Create subplot and plot
    fig, ax = plt.subplots()
    for festividad, data in df.groupby('FESTIVIDAD'):
        ax.plot(data.index, data['PRODUCTOS QUE SE VENDERÁN'],
                label='Festividad: {}'.format("No" if festividad == 0 else "Sí"),
                color=festividad_palette[festividad])
    # Set Title
    ax.set_title('Productos vendidos por Mes', fontweight="bold")
    # Set xlabel and ylabel
    ax.set_xlabel('Mes')
    ax.set_ylabel('Productos vendidos')
    # Set xticks to be the index of the dataframe
    ax.set_xticks(df.index)
    # Set xticklabels to be the months
    ax.set_xticklabels(df['MES'])
    # Add legend
    ax.legend()
    # Save the plot as a PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)


def generate_matplotlib_scatter_plot(df, filename):
    # Define color
    color = '#00E0B7'  # Color personalizado
    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df['PRODUCTOS ALMACENADOS'], df['PRODUCTOS QUE SE VENDERÁN'], color=color,
               alpha=0.7)  # Utiliza el color personalizado y reduce la opacidad
    # Set Title
    ax.set_title('Productos almacenados vs productos vendidos', fontweight="bold")
    # Set xlabel and ylabel
    ax.set_xlabel('PRODUCTOS ALMACENADOS')
    ax.set_ylabel('PRODUCTOS QUE SE VENDERÁN')
    # Save the plot as a PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)


# Utilizando las funciones proporcionadas
generate_matplotlib_line_chart(df_con_predicciones,
                               'images/productos_vendidos_por_mes_y_festividad.png')
generate_matplotlib_scatter_plot(df_con_predicciones,
                                 'images/dispersion_productos_almacenados_vs_vendidos.png')


def generate_matplotlib_scatter_plot(df, filename):
    # Create subplot
    fig, ax = plt.subplots()
    # Scatter plot
    ax.scatter(df['PRODUCTOS ALMACENADOS'], df['PRODUCTOS QUE SE VENDERÁN'], color="#F4A261", alpha=0.8)
    # Set Title
    ax.set_title('Gráfico de Dispersión: Productos Vendidos vs Productos Almacenados', fontweight="bold")
    # Set xlabel
    ax.set_xlabel('Productos Almacenados')
    # Set ylabel
    ax.set_ylabel('Productos Que Se Venderán')
    # Save the plot as a PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def generate_profit_plot(df, filename):
    # Calculate profit
    df['Ganancias'] = df['PRODUCTOS QUE SE VENDERÁN'] * df['PRECIO DE VENTA'] - (
            df['GASTO DE MARKETING'] + df['GASTO DE ALMACENAMIENTO'])
    # Create subplot
    fig, ax = plt.subplots()
    # Bar plot
    ax.bar(df.index, df['Ganancias'], color="#FF5733", label='Ganancias')  # Gráfico de barras para ganancias
    # Line plot for better visualization of profit trend
    ax.plot(df.index, df['Ganancias'], color="#0066FF", marker='o', linestyle='-', linewidth=2,
            label='Ganancias (Tendencia)')  # Gráfico de línea para tendencia de ganancias
    # Set Title
    ax.set_title('Gráfico de Ganancias por Mes', fontweight="bold")
    # Set xlabel
    ax.set_xlabel('MES')
    # Set xticks to be the index of the dataframe
    ax.set_xticks(df.index)
    # Set xticklabels to be the months
    ax.set_xticklabels(df['MES'])
    # Set ylabel
    ax.set_ylabel('Ganancias ($)')
    # Add legend
    ax.legend()
    # Save the plot as a PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)


generate_matplotlib_scatter_plot(df_con_predicciones, 'images/dispersion_productos_vendidos_vs_almacenados.png')
generate_profit_plot(df_con_predicciones, 'images/ganancias_mensuales.png')


# Crear una clase personalizada que herede de FPDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Resultados de Predicciones', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')


def create_letterhead(pdf, width, height):
    # Añadir el membrete
    pdf.image("images/logo.png", 0, 0, width)

    # Ajustar la posición vertical para que el título aparezca después del membrete
    pdf.set_y(height / 4)


def create_title(title, pdf):
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)

    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128, g=128, b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')

    # Add line break
    pdf.ln(10)


def write_to_pdf(pdf, words):
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.set_font('Helvetica', '', 12)

    pdf.write(5, words)


# Función para crear el PDF
def create_pdf():
    title = "Resultados de Predicciones"
    width = 210
    height = 297
    # Create PDF
    pdf = PDF()  # A4 (210 by 297 mm)
    pdf.add_page()

    # Add lettterhead and title
    create_letterhead(pdf, width, height)
    pdf.ln(10)
    create_title(title, pdf)

    # Add some words to PDF
    write_to_pdf(pdf, "1. La tabla a continuación ilustra las ventas mensuales del Inventario Predichas:")
    pdf.ln(15)

    # Add table
    pdf.image("images/ventas_mensuales.png", w=170)
    pdf.ln(10)

    write_to_pdf(pdf, "Se observan barras en las celdas de la columna donde se visualiza de mejor manera "
                      "la cantidad de productos que se venderán")

    '''
    Second Page of PDF
    '''

    # Add Page
    pdf.add_page()

    # Add some words to PDF
    write_to_pdf(pdf,
                 "2. Las visualizaciones a continuación muestran la tendencia de las ventas por mes ")
    pdf.ln(15)
    # Add the generated visualisations to the PDF
    pdf.image("images/productos_vendidos_por_mes_y_festividad.png", 10, 30, width / 2 - 15)
    pdf.image("images/dispersion_productos_almacenados_vs_vendidos.png", width / 2 + 10, 30, width / 2 - 15)

    pdf.ln(70)
    # Add some words to PDF
    write_to_pdf(pdf,
                 "3.  Las visualizaciones a continuación muestran el pronostico de las futuras "
                 "ganancias mensuales")
    pdf.ln(15)
    pdf.image("images/dispersion_productos_vendidos_vs_almacenados.png", 10, 130, width / 2 - 15)
    pdf.image("images/ganancias_mensuales.png", width / 2 + 10, 130, width / 2 - 15)
    # Guardar el PDF
    pdf_filename = "resultados_prediccion.pdf"
    pdf.output(pdf_filename)

    return pdf_filename


# Botón para descargar PDF
if st.button("Descargar Resultados como PDF"):
    # Generar el PDF
    pdf_filename = create_pdf()
    st.success(f"El reporte se ha descargado como {pdf_filename}")

    # Convertir el PDF en bytes para descargarlo
    with open(pdf_filename, "rb") as f:
        pdf_bytes = f.read()

    # Codificar el PDF en base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="resultados_prediccion.pdf">Descargar PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
