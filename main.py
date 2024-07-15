import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configurar la página (debe ser el primer comando de Streamlit)
st.set_page_config(page_title='Supermarket Sales Analysis', layout='wide')

# Crear una carpeta temporal para almacenar las imágenes
if not os.path.exists("temp_images"):
    os.makedirs("temp_images")

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('supermarket_sales.csv', sep=';')
    
    # Convertir la columna 'Total' a numérica
    if df['Total'].dtype == 'object':
        df['Total'] = df['Total'].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Convertir la columna 'cogs' a numérica si es de tipo string
    if df['cogs'].dtype == 'object':
        df['cogs'] = df['cogs'].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Convertir la columna 'gross income' a numérica si es de tipo string
    if df['gross income'].dtype == 'object':
        df['gross income'] = df['gross income'].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Convertir la columna 'Date' a tipo datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Crear la columna 'Month'
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['Day of Week'] = df['Date'].dt.day_name()
    df['Hour'] = df['Time'].str[:2].astype(int)
    
    return df

df = load_data()

st.title('Análisis de Ventas de Supermercado')

# Función para guardar gráficos
def save_plot(fig, filename):
    fig.savefig(f"temp_images/{filename}.png")

# Función para crear y mostrar gráficos
def create_and_show_plot(data, x, y, title, xlabel, ylabel, plot_type='bar', rotation=0, figsize=(8, 4), hue=None):
    fig, ax = plt.subplots(figsize=figsize)
    if plot_type == 'bar':
        sns.barplot(data=data, x=x, y=y, ax=ax, hue=hue)
    elif plot_type == 'line':
        sns.lineplot(data=data, x=x, y=y, marker='o', ax=ax, hue=hue)
    elif plot_type == 'hist':
        sns.histplot(data[x], bins=10, kde=True, ax=ax)
    elif plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, ax=ax, hue=hue)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    st.pyplot(fig)
    save_plot(fig, title.replace(' ', '_').lower())

# Lista de visualizaciones
visualizations = [
    ('Ventas Totales por Ciudad', 'City', 'Total', 'Ventas Totales por Ciudad', 'Ciudad', 'Ventas Totales'),
    ('Ventas Totales por Género', 'Gender', 'Total', 'Ventas Totales por Género', 'Género', 'Ventas Totales'),
    ('Ventas Totales por Línea de Producto', 'Product line', 'Total', 'Ventas Totales por Línea de Producto', 'Línea de Producto', 'Ventas Totales', 45),
    ('Ingresos por Método de Pago', 'Payment', 'Total', 'Ingresos por Método de Pago', 'Método de Pago', 'Ingresos Totales'),
    ('Ventas Mensuales', 'Month', 'Total', 'Ventas Mensuales', 'Mes', 'Ventas Totales', 45, 'line'),
    ('Distribución de Calificaciones', 'Rating', None, 'Distribución de Calificaciones', 'Calificación', 'Frecuencia', 0, 'hist'),
    ('Ingreso Bruto por Calificación', 'Rating', 'gross income', 'Ingreso Bruto por Calificación', 'Calificación', 'Ingreso Bruto'),
    ('Ventas por Tipo de Cliente', 'Customer type', 'Total', 'Ventas por Tipo de Cliente', 'Tipo de Cliente', 'Ventas Totales'),
    ('Cantidad de Productos Vendidos por Ciudad', 'City', 'Quantity', 'Cantidad de Productos Vendidos por Ciudad', 'Ciudad', 'Cantidad de Productos Vendidos'),
    ('Cantidad de Productos Vendidos por Género', 'Gender', 'Quantity', 'Cantidad de Productos Vendidos por Género', 'Género', 'Cantidad de Productos Vendidos'),
    ('Cantidad de Productos Vendidos por Línea de Producto', 'Product line', 'Quantity', 'Cantidad de Productos Vendidos por Línea de Producto', 'Línea de Producto', 'Cantidad de Productos Vendidos', 45),
    ('Promedio de Calificaciones por Ciudad', 'City', 'Rating', 'Promedio de Calificaciones por Ciudad', 'Ciudad', 'Calificación Promedio'),
    ('Promedio de Calificaciones por Género', 'Gender', 'Rating', 'Promedio de Calificaciones por Género', 'Género', 'Calificación Promedio'),
    ('Ventas por Día de la Semana', 'Day of Week', 'Total', 'Ventas por Día de la Semana', 'Día de la Semana', 'Ventas Totales', 45),
    ('Ventas por Hora del Día', 'Hour', 'Total', 'Ventas por Hora del Día', 'Hora del Día', 'Ventas Totales', 0, 'line'),
    ('Ingresos Totales por Mes', 'Month', 'Total', 'Ingresos Totales por Mes', 'Mes', 'Ingresos Totales', 45),
    ('Relación entre Cantidad de Productos y Total de Ventas', 'Quantity', 'Total', 'Relación entre Cantidad de Productos y Total de Ventas', 'Cantidad de Productos', 'Total de Ventas', 0, 'scatter'),
    ('Relación entre Precio Unitario y Cantidad Vendida', 'Unit price', 'Quantity', 'Relación entre Precio Unitario y Cantidad Vendida', 'Precio Unitario', 'Cantidad Vendida', 0, 'scatter'),
    ('Promedio de Ingresos por Tipo de Cliente', 'Customer type', 'Total', 'Promedio de Ingresos por Tipo de Cliente', 'Tipo de Cliente', 'Ingreso Promedio'),
    ('Promedio de Ingresos por Método de Pago', 'Payment', 'Total', 'Promedio de Ingresos por Método de Pago', 'Método de Pago', 'Ingreso Promedio'),
    ('Distribución de Ventas por Sucursal', 'Branch', 'Total', 'Distribución de Ventas por Sucursal', 'Sucursal', 'Ventas Totales'),
    ('Ingresos por Sucursal', 'Branch', 'Total', 'Ingresos por Sucursal', 'Sucursal', 'Ingresos Totales'),
    ('Ventas por Línea de Producto en Cada Ciudad', 'City', 'Total', 'Ventas por Línea de Producto en Cada Ciudad', 'Ciudad', 'Ventas Totales', 0, 'bar', 'Product line'),
    ('Ventas por Género en Cada Ciudad', 'City', 'Total', 'Ventas por Género en Cada Ciudad', 'Ciudad', 'Ventas Totales', 0, 'bar', 'Gender'),
    ('Cantidad de Productos Vendidos por Método de Pago', 'Payment', 'Quantity', 'Cantidad de Productos Vendidos por Método de Pago', 'Método de Pago', 'Cantidad de Productos Vendidos'),
    ('Distribución de Clientes por Tipo', 'Customer type', 'Count', 'Distribución de Clientes por Tipo', 'Tipo de Cliente', 'Cantidad de Clientes'),
    ('Calificaciones por Línea de Producto', 'Product line', 'Rating', 'Calificaciones por Línea de Producto', 'Línea de Producto', 'Calificación Promedio', 45),
    ('Ingresos por Tipo de Cliente y Género', 'Customer type', 'Total', 'Ingresos por Tipo de Cliente y Género', 'Tipo de Cliente', 'Ingresos Totales', 0, 'bar', 'Gender'),
    ('Ingresos por Sucursal y Método de Pago', 'Branch', 'Total', 'Ingresos por Sucursal y Método de Pago', 'Sucursal', 'Ingresos Totales', 0, 'bar', 'Payment'),
    ('Relación entre Ingresos y Tiempo de Compra', 'Time', 'Total', 'Relación entre Ingresos y Tiempo de Compra', 'Hora de Compra', 'Ingresos Totales', 45, 'scatter', 'Branch')
]

# Generar visualizaciones
for viz in visualizations:
    title, x, y, plot_title, xlabel, ylabel = viz[:6]
    rotation = viz[6] if len(viz) > 6 else 0
    plot_type = viz[7] if len(viz) > 7 else 'bar'
    hue = viz[8] if len(viz) > 8 else None
    st.subheader(title)
    if y:
        data = df.groupby(x)[y].sum().reset_index()
    else:
        data = df
    create_and_show_plot(data, x, y, plot_title, xlabel, ylabel, plot_type, rotation, hue=hue)

# Función para generar el PDF
def generate_pdf():
    c = canvas.Canvas("report.pdf", pagesize=letter)
    width, height = letter
    y = height - 40
    
    for img in os.listdir("temp_images"):
        if img.endswith(".png"):
            img_path = os.path.join("temp_images", img)
            c.drawImage(ImageReader(img_path), 40, y - 400, width - 80, 400)
            y -= 450
            if y < 100:
                c.showPage()
                y = height - 40
    
    c.save()

# Botón para generar el informe PDF
if st.button('Generar Informe PDF'):
    generate_pdf()
    st.success('Informe PDF generado con éxito. Puedes descargarlo [aquí](report.pdf)')