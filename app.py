import streamlit as st

# Configurar la página (debe ser el primer comando de Streamlit)
st.set_page_config(page_title='Supermarket Sales Analysis', layout='wide')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    return df

df = load_data()

st.title('Análisis de Ventas de Supermercado')

# Función para guardar gráficos
def save_plot(fig, filename):
    fig.savefig(f"temp_images/{filename}.png")

# Ventas totales por ciudad
st.subheader('Ventas Totales por Ciudad')
ventas_por_ciudad = df.groupby('City')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_ciudad, x='City', y='Total', ax=ax)
plt.title('Ventas Totales por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Ventas Totales')
st.pyplot(fig)

# Ventas totales por género
st.subheader('Ventas Totales por Género')
ventas_por_genero = df.groupby('Gender')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_genero, x='Gender', y='Total', ax=ax)
plt.title('Ventas Totales por Género')
plt.xlabel('Género')
plt.ylabel('Ventas Totales')
st.pyplot(fig)

# Ventas totales por línea de producto
st.subheader('Ventas Totales por Línea de Producto')
ventas_por_producto = df.groupby('Product line')['Total'].sum().reset_index()
ventas_por_producto = ventas_por_producto.sort_values(by='Total', ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_producto, x='Product line', y='Total', ax=ax)
plt.title('Ventas Totales por Línea de Producto')
plt.xlabel('Línea de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
st.pyplot(fig)

# Ingresos por método de pago
st.subheader('Ingresos por Método de Pago')
ingresos_por_pago = df.groupby('Payment')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ingresos_por_pago, x='Payment', y='Total', ax=ax)
plt.title('Ingresos por Método de Pago')
plt.xlabel('Método de Pago')
plt.ylabel('Ingresos Totales')
st.pyplot(fig)

# Ventas mensuales
st.subheader('Ventas Mensuales')
df['Month'] = df['Date'].dt.to_period('M').astype(str)
ventas_mensuales = df.groupby('Month')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.lineplot(data=ventas_mensuales, x='Month', y='Total', marker='o', ax=ax)
plt.title('Ventas Mensuales')
plt.xlabel('Mes')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
st.pyplot(fig)

# Calificaciones de clientes
st.subheader('Distribución de Calificaciones')
fig, ax = plt.subplots()
sns.histplot(df['Rating'], bins=10, kde=True, ax=ax)
plt.title('Distribución de Calificaciones')
plt.xlabel('Calificación')
plt.ylabel('Frecuencia')
st.pyplot(fig)

# Ingreso bruto por calificación
st.subheader('Ingreso Bruto por Calificación')
ingreso_bruto_calificacion = df.groupby('Rating')['gross income'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ingreso_bruto_calificacion, x='Rating', y='gross income', ax=ax)
plt.title('Ingreso Bruto por Calificación')
plt.xlabel('Calificación')
plt.ylabel('Ingreso Bruto')
st.pyplot(fig)

# Ventas por tipo de cliente
st.subheader('Ventas por Tipo de Cliente')
ventas_por_cliente = df.groupby('Customer type')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_cliente, x='Customer type', y='Total', ax=ax)
plt.title('Ventas por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Ventas Totales')
st.pyplot(fig)

# Cantidad de productos vendidos por ciudad
st.subheader('Cantidad de Productos Vendidos por Ciudad')
productos_por_ciudad = df.groupby('City')['Quantity'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=productos_por_ciudad, x='City', y='Quantity', ax=ax)
plt.title('Cantidad de Productos Vendidos por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Cantidad de Productos Vendidos')
st.pyplot(fig)

# Cantidad de productos vendidos por género
st.subheader('Cantidad de Productos Vendidos por Género')
productos_por_genero = df.groupby('Gender')['Quantity'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=productos_por_genero, x='Gender', y='Quantity', ax=ax)
plt.title('Cantidad de Productos Vendidos por Género')
plt.xlabel('Género')
plt.ylabel('Cantidad de Productos Vendidos')
st.pyplot(fig)

# Cantidad de productos vendidos por línea de producto
st.subheader('Cantidad de Productos Vendidos por Línea de Producto')
productos_por_producto = df.groupby('Product line')['Quantity'].sum().reset_index()
productos_por_producto = productos_por_producto.sort_values(by='Quantity', ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=productos_por_producto, x='Product line', y='Quantity', ax=ax)
plt.title('Cantidad de Productos Vendidos por Línea de Producto')
plt.xlabel('Línea de Producto')
plt.ylabel('Cantidad de Productos Vendidos')
plt.xticks(rotation=45)
st.pyplot(fig)

# Promedio de calificaciones por ciudad
st.subheader('Promedio de Calificaciones por Ciudad')
calificacion_prom_ciudad = df.groupby('City')['Rating'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=calificacion_prom_ciudad, x='City', y='Rating', ax=ax)
plt.title('Promedio de Calificaciones por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Calificación Promedio')
st.pyplot(fig)

# Promedio de calificaciones por género
st.subheader('Promedio de Calificaciones por Género')
calificacion_prom_genero = df.groupby('Gender')['Rating'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=calificacion_prom_genero, x='Gender', y='Rating', ax=ax)
plt.title('Promedio de Calificaciones por Género')
plt.xlabel('Género')
plt.ylabel('Calificación Promedio')
st.pyplot(fig)

# Ventas por día de la semana
st.subheader('Ventas por Día de la Semana')
df['Day of Week'] = df['Date'].dt.day_name()
ventas_por_dia_semana = df.groupby('Day of Week')['Total'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_dia_semana, x='Day of Week', y='Total', ax=ax)
plt.title('Ventas por Día de la Semana')
plt.xlabel('Día de la Semana')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
st.pyplot(fig)

# Ventas por hora del día
st.subheader('Ventas por Hora del Día')
df['Hour'] = df['Time'].str[:2].astype(int)
ventas_por_hora = df.groupby('Hour')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.lineplot(data=ventas_por_hora, x='Hour', y='Total', marker='o', ax=ax)
plt.title('Ventas por Hora del Día')
plt.xlabel('Hora del Día')
plt.ylabel('Ventas Totales')
st.pyplot(fig)

# Ingresos totales por mes
st.subheader('Ingresos Totales por Mes')
ingresos_totales_mes = df.groupby('Month')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ingresos_totales_mes, x='Month', y='Total', ax=ax)
plt.title('Ingresos Totales por Mes')
plt.xlabel('Mes')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)
st.pyplot(fig)

# Relación entre cantidad de productos y total de ventas
st.subheader('Relación entre Cantidad de Productos y Total de Ventas')
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Quantity', y='Total', ax=ax)
plt.title('Relación entre Cantidad de Productos y Total de Ventas')
plt.xlabel('Cantidad de Productos')
plt.ylabel('Total de Ventas')
st.pyplot(fig)

# Relación entre precio unitario y cantidad vendida
st.subheader('Relación entre Precio Unitario y Cantidad Vendida')
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Unit price', y='Quantity', ax=ax)
plt.title('Relación entre Precio Unitario y Cantidad Vendida')
plt.xlabel('Precio Unitario')
plt.ylabel('Cantidad Vendida')
st.pyplot(fig)

# Promedio de ingresos por tipo de cliente
st.subheader('Promedio de Ingresos por Tipo de Cliente')
promedio_ingresos_cliente = df.groupby('Customer type')['Total'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=promedio_ingresos_cliente, x='Customer type', y='Total', ax=ax)
plt.title('Promedio de Ingresos por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Ingreso Promedio')
st.pyplot(fig)

# Promedio de ingresos por método de pago
st.subheader('Promedio de Ingresos por Método de Pago')
promedio_ingresos_pago = df.groupby('Payment')['Total'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=promedio_ingresos_pago, x='Payment', y='Total', ax=ax)
plt.title('Promedio de Ingresos por Método de Pago')
plt.xlabel('Método de Pago')
plt.ylabel('Ingreso Promedio')
st.pyplot(fig)

# Distribución de ventas por sucursal
st.subheader('Distribución de Ventas por Sucursal')
ventas_por_sucursal = df.groupby('Branch')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ventas_por_sucursal, x='Branch', y='Total', ax=ax)
plt.title('Distribución de Ventas por Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Ventas Totales')
st.pyplot(fig)

# Ingresos por sucursal
st.subheader('Ingresos por Sucursal')
ingresos_por_sucursal = df.groupby('Branch')['Total'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=ingresos_por_sucursal, x='Branch', y='Total', ax=ax)
plt.title('Ingresos por Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Ingresos Totales')
st.pyplot(fig)

# Ventas por línea de producto en cada ciudad
st.subheader('Ventas por Línea de Producto en Cada Ciudad')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df, x='City', y='Total', hue='Product line', ci=None, ax=ax)
plt.title('Ventas por Línea de Producto en Cada Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Ventas Totales')
plt.legend(title='Línea de Producto')
st.pyplot(fig)

# Ventas por género en cada ciudad
st.subheader('Ventas por Género en Cada Ciudad')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df, x='City', y='Total', hue='Gender', ci=None, ax=ax)
plt.title('Ventas por Género en Cada Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Ventas Totales')
plt.legend(title='Género')
st.pyplot(fig)

# Cantidad de productos vendidos por método de pago
st.subheader('Cantidad de Productos Vendidos por Método de Pago')
productos_por_pago = df.groupby('Payment')['Quantity'].sum().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=productos_por_pago, x='Payment', y='Quantity', ax=ax)
plt.title('Cantidad de Productos Vendidos por Método de Pago')
plt.xlabel('Método de Pago')
plt.ylabel('Cantidad de Productos Vendidos')
st.pyplot(fig)

# Distribución de clientes por tipo
st.subheader('Distribución de Clientes por Tipo')
clientes_por_tipo = df['Customer type'].value_counts().reset_index()
clientes_por_tipo.columns = ['Customer type', 'Count']
fig, ax = plt.subplots()
sns.barplot(data=clientes_por_tipo, x='Customer type', y='Count', ax=ax)
plt.title('Distribución de Clientes por Tipo')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Cantidad de Clientes')
st.pyplot(fig)

# Calificaciones por línea de producto
st.subheader('Calificaciones por Línea de Producto')
calificaciones_por_producto = df.groupby('Product line')['Rating'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=calificaciones_por_producto, x='Product line', y='Rating', ax=ax)
plt.title('Calificaciones por Línea de Producto')
plt.xlabel('Línea de Producto')
plt.ylabel('Calificación Promedio')
plt.xticks(rotation=45)
st.pyplot(fig)

# Ingresos por tipo de cliente y género
st.subheader('Ingresos por Tipo de Cliente y Género')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df, x='Customer type', y='Total', hue='Gender', ci=None, ax=ax)
plt.title('Ingresos por Tipo de Cliente y Género')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Ingresos Totales')
plt.legend(title='Género')
st.pyplot(fig)

# Ingresos por sucursal y método de pago
st.subheader('Ingresos por Sucursal y Método de Pago')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df, x='Branch', y='Total', hue='Payment', ci=None, ax=ax)
plt.title('Ingresos por Sucursal y Método de Pago')
plt.xlabel('Sucursal')
plt.ylabel('Ingresos Totales')
plt.legend(title='Método de Pago')
st.pyplot(fig)

# Relación entre ingresos y tiempo de compra
st.subheader('Relación entre Ingresos y Tiempo de Compra')
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df, x='Time', y='Total', hue='Branch', ax=ax)
plt.title('Relación entre Ingresos y Tiempo de Compra')
plt.xlabel('Hora de Compra')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)
st.pyplot(fig)
