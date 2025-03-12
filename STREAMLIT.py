import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Agregar estilos personalizados
def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp { background-color: #FFFFFF; }
        .css-1d391kg { background-color: #F0F2F6 !important; }
        .stButton>button {
            background-color: #00B140;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
        }
        .stTextInput>div>div>input { color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Cargar datos con cache para mejorar el rendimiento
@st.cache_data
def load_data():
    return pd.read_csv("niqfinal.csv")

df = load_data()

# Sidebar - Filtros de datos dinámicos
def sidebar_filters(df):
    st.sidebar.header("Filtros de Datos")
    market = st.sidebar.selectbox("Selecciona un Market", ["Todos"] + df["Markets"].dropna().unique().tolist())
    canasto = st.sidebar.selectbox("Selecciona un Canasto", ["Todos"] + df["CANASTO"].dropna().unique().tolist())
    categoria = st.sidebar.selectbox("Selecciona una Categoría", ["Todos"] + df["CATEGORIA"].dropna().unique().tolist())
    
    if categoria != "Todos":
        segmentos_disponibles = df[df["CATEGORIA"] == categoria]["M.SEGMENTO"].dropna().unique().tolist()
    else:
        segmentos_disponibles = df["M.SEGMENTO"].dropna().unique().tolist()
    
    segmento = st.sidebar.selectbox("Selecciona un Segmento", ["Todos"] + segmentos_disponibles)
    return market, canasto, categoria, segmento

# Aplicar filtros
market, canasto, categoria, segmento = sidebar_filters(df)

def filter_data(df, market, canasto, categoria, segmento):
    if market != "Todos":
        df = df[df["Markets"] == market]
    if canasto != "Todos":
        df = df[df["CANASTO"] == canasto]
    if categoria != "Todos":
        df = df[df["CATEGORIA"] == categoria]
    if segmento != "Todos":
        df = df[df["M.SEGMENTO"] == segmento]
    return df

df_filtered = filter_data(df, market, canasto, categoria, segmento)

st.title("Análisis y Predicción de Ventas")
st.write("## Información General")
st.write(f"Total de filas: {df_filtered.shape[0]}")
st.write(df_filtered.head())

if df_filtered.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    def plot_sales_trend(df):
        if "Periods" in df.columns and "Vtas Valor" in df.columns:
            df["Periods"] = pd.to_datetime(df["Periods"], errors='coerce')
            df_grouped = df.groupby("Periods")["Vtas Valor"].sum().reset_index()
            plt.figure(figsize=(10, 5))
            sns.lineplot(x="Periods", y="Vtas Valor", data=df_grouped, marker="o")
            plt.xticks(rotation=90)
            plt.title("Tendencia de Ventas en Valor")
            plt.xlabel("Periodo")
            plt.ylabel("Ventas en Valor")
            st.pyplot(plt)
            plt.close()
    
    st.write("## Tendencias de Ventas")
    plot_sales_trend(df_filtered)
    
    def plot_price_distribution(df):
        if "Precio EQ2 Promedio" in df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df["Precio EQ2 Promedio"].dropna(), bins=30, kde=True)
            plt.title("Distribución de Precios EQ2")
            st.pyplot(plt)
            plt.close()
    
    st.write("## Distribución de Precios")
    plot_price_distribution(df_filtered)
    

    
    ventas_totales = df_filtered["Vtas Valor"].sum() if "Vtas Valor" in df_filtered.columns else 0
    precio_promedio = df_filtered["Precio EQ2 Promedio"].mean() if "Precio EQ2 Promedio" in df_filtered.columns else 0
    st.metric(label="Ventas Totales ($)", value=f"{ventas_totales:,.2f}")
    st.metric(label="Precio Promedio EQ2 ($)", value=f"{precio_promedio:,.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima

def predict_sales(df, column, key):
    if "Periods" not in df.columns or column not in df.columns:
        st.error("La columna de períodos o la de ventas no están en los datos.")
        return
    
    # Input del usuario con clave única
    variation = st.number_input(
        "Ingresa el porcentaje de variación para escenarios optimista y pesimista", 
        min_value=0.0, max_value=100.0, value=5.0, key=key
    ) / 100
    
    df_time_series = df[["Periods", column]].groupby("Periods").sum().reset_index()
    df_time_series["Periods"] = pd.to_datetime(df_time_series["Periods"], errors='coerce')
    df_time_series = df_time_series.dropna().sort_values("Periods")
    
    # Modelo AutoARIMA
    model = auto_arima(df_time_series[column], seasonal=True, m=12, trace=False, suppress_warnings=True)
    future_periods = 5
    forecast = model.predict(n_periods=future_periods)
    
    # Crear fechas futuras
    last_date = df_time_series["Periods"].max()
    future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq='M')[1:]
    
    # Crear DataFrame con los escenarios
    forecast_df = pd.DataFrame({
        "Periods": future_dates,
        "Base": forecast,
        "Optimista": forecast * (1 + variation),
        "Pesimista": forecast * (1 - variation)
    })
    
    # Graficar los escenarios
    plt.figure(figsize=(10, 5))
    plt.plot(df_time_series["Periods"], df_time_series[column], marker="o", label="Histórico", color='black')
    plt.plot(forecast_df["Periods"], forecast_df["Base"], marker="o", linestyle="dashed", label="Escenario Base", color='blue')
    plt.plot(forecast_df["Periods"], forecast_df["Optimista"], marker="o", linestyle="dashed", label="Escenario Optimista", color='green')
    plt.plot(forecast_df["Periods"], forecast_df["Pesimista"], marker="o", linestyle="dashed", label="Escenario Pesimista", color='red')
    
    plt.xticks(rotation=45)
    plt.xlabel("Periodo")
    plt.ylabel(column)
    plt.title(f"Predicción de {column} con escenarios")
    plt.legend()
    st.pyplot(plt)
    
    st.write(forecast_df)

st.write("## Predicción de Ventas en Valor")
predict_sales(df_filtered, "Vtas Valor", key="ventas_valor")

st.write("## Predicción de Ventas en Unidades")
predict_sales(df_filtered, "Vtas Unds", key="ventas_unidades")

def predict_flavor_sales(df, sales_column, key_prefix):
    if "Periods" not in df.columns or "M.SABOR" not in df.columns or sales_column not in df.columns:
        st.error("Las columnas requeridas no están en los datos.")
        return
    
    # Convertir la fecha a formato datetime
    df["Periods"] = pd.to_datetime(df["Periods"], errors='coerce')
    
    # Selección de sabor
    flavor_list = df["M.SABOR"].unique().tolist()
    selected_flavor = st.selectbox("Selecciona un sabor", flavor_list, key=f"{key_prefix}_flavor")
    
    # Filtrar datos del sabor seleccionado
    df_flavor = df[df["M.SABOR"] == selected_flavor].dropna().sort_values("Periods")
    
    if df_flavor.empty:
        st.warning("No hay datos suficientes para este sabor.")
        return
    
    # Input del usuario para variación
    variation = st.number_input(
        f"Ingrese el porcentaje de variación para {selected_flavor}",
        min_value=0.0, max_value=100.0, value=5.0, key=f"{key_prefix}_variation"
    ) / 100
    
    # Aplicar modelo AutoARIMA
    model = auto_arima(df_flavor[sales_column], seasonal=True, m=12, trace=False, suppress_warnings=True)
    future_periods = 5
    forecast = model.predict(n_periods=future_periods)
    
    # Crear fechas futuras
    last_date = df_flavor["Periods"].max()
    future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq='M')[1:]
    
    # Crear DataFrame con predicciones
    forecast_df = pd.DataFrame({
        "Periods": future_dates,
        "Base": forecast,
        "Optimista": forecast * (1 + variation),
        "Pesimista": forecast * (1 - variation)
    })
    
    # Graficar la predicción
    plt.figure(figsize=(8, 4))
    plt.plot(df_flavor["Periods"], df_flavor[sales_column], marker="o", label="Histórico", color='black')
    plt.plot(forecast_df["Periods"], forecast_df["Base"], marker="o", linestyle="dashed", label="Base", color='blue')
    plt.plot(forecast_df["Periods"], forecast_df["Optimista"], marker="o", linestyle="dashed", label="Optimista", color='green')
    plt.plot(forecast_df["Periods"], forecast_df["Pesimista"], marker="o", linestyle="dashed", label="Pesimista", color='red')
    
    plt.xticks(rotation=45)
    plt.xlabel("Periodo")
    plt.ylabel(sales_column)
    plt.title(f"Predicción de {sales_column} para {selected_flavor}")
    plt.legend()
    st.pyplot(plt)
    
    st.write(forecast_df)

# Predicciones para Ventas en Valor y Ventas en Unidades
st.write("## Predicción de Ventas por Sabor")
predict_flavor_sales(df_filtered, "Vtas Valor", key_prefix="flavor_valor")
predict_flavor_sales(df_filtered, "Vtas Unds", key_prefix="flavor_unidades")