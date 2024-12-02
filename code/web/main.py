# Imports
import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import folium
from streamlit_folium import st_folium
import pickle

# -- Funciones para cargar datos -- #
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv', sep=',')
    return df
@st.cache_resource
def load_label_encoders():
    with open('le_day_of_week.pkl', 'rb') as file:
        le_day_of_week = pickle.load(file)
    with open('le_provincia_destino_name.pkl', 'rb') as file:
        le_provincia_destino_name = pickle.load(file)
    return le_day_of_week, le_provincia_destino_name
# -- -- #

# --- Configuración de la página ---
st.set_page_config(
    page_title="DataNomads",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cargar logotipos y convertirlos a base64 --- #
datanomads_logo_path = "datanomads_logo.png"
telefonica_logo_path = "telefonica_logo.png"

def get_base64_encoded_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"No se encontró el archivo del logotipo en: {image_path}")
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Codificar logotipos
datanomads_logo_base64 = get_base64_encoded_image(datanomads_logo_path)
telefonica_logo_base64 = get_base64_encoded_image(telefonica_logo_path)

# --- Estilos personalizados --- #
st.markdown(
    f"""
    <style>
    .main {{
        background-color: #f4f4f4; /* Fondo gris claro */
        color: #333333; /* Texto gris oscuro */
    }}
    .sidebar .sidebar-content {{
        background-color: #e9ecef; /* Fondo de la barra lateral */
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #0073c8; /* Azul Telefónica */
    }}
    .stButton > button {{
        background-color: #0073c8; /* Azul Telefónica */
        color: white;
        border-radius: 5px;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: #005b9f;
        color: white;
    }}
    /* Contenedor de encabezado */
    .header-container {{
        position: relative;
        padding: 10px 0;
        border-bottom: 2px solid #ddd; /* Línea divisoria opcional */
    }}
    .header-container .logo-left {{
        position: absolute;
        top: 10px; /* Ajusta la altura del logo de DataNomads */
        left: 10px; /* Ajusta la distancia del borde izquierdo */
        width: 100px; /* Tamaño ajustado del logo de DataNomads */
        height: auto;
    }}
    .header-container .title {{
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        color: #0073c8; /* Azul Telefónica */
        margin-top: 100px; /* Espaciado para que el título esté debajo del logo */
    }}
    .header-container .logo-right {{
        position: absolute;
        top: 10px; /* Ajusta la altura del logo de Telefónica */
        right: 10px; /* Ajusta la distancia del borde derecho */
        width: 80px; /* Tamaño ajustado del logo de Telefónica */
        height: auto;
    }}
    /* Cambiar el contorno de los selectbox */
    .stSelectbox div[data-baseweb="select"] > div {{
        border: 2px solid #d9d9d9; /* Color gris predeterminado */
        border-radius: 5px;
        transition: border-color 0.2s ease;
    }}
    .stSelectbox div[data-baseweb="select"]:focus-within > div {{
        border: 2px solid #0073c8 !important; /* Azul Telefónica al enfocarse */
        border-radius: 5px;
    }}
    </style>
    <div class="header-container">
        <img src="data:image/png;base64,{datanomads_logo_base64}" alt="DataNomads Logo" class="logo-left">
        <div class="title">DataNomads</div>
        <img src="data:image/png;base64,{telefonica_logo_base64}" alt="Telefónica Logo" class="logo-right">
    </div>
    """,
    unsafe_allow_html=True
)


# --- Cargar datos --- #
model = load_model()
le_day_of_week, le_provincia_destino_name = load_label_encoders()
og_data = load_data()


# --- Inicializar session_state ---
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "selected_province" not in st.session_state:
    st.session_state.selected_province = None

# --- Diccionario de Mapeo para Provincias ---
province_name_mapping = {
    "Illes Balears": "Balears, Illes",  # Ajuste del nombre devuelto por el mapa
    "Las Palmas": "Palmas, Las",
    "Santa Cruz De Tenerife": "Santa Cruz de Tenerife",  # Ajuste de la capitalización
    # Agrega más mapeos si fuera necesario
}

# --- Provincias disponibles en los datos ---
available_provinces = {"Balears, Illes", "Palmas, Las", "Santa Cruz de Tenerife"}

# --- Función para resetear la predicción cuando se cambian valores ---
def reset_prediction():
    st.session_state.prediction = None
    st.session_state.show_prediction = False

# --- Mapa Interactivo de Provincias ---
st.markdown("## Mapa Interactivo de Provincias de España")

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-provinces.geojson"
    import requests
    response = requests.get(url)
    return response.json()

geojson_data = load_geojson()

def style_function(feature):
    selected_province = st.session_state.selected_province
    if selected_province == feature["properties"]["name"]:
        return {"fillColor": "red", "color": "black", "weight": 2, "fillOpacity": 0.6}
    return {"fillColor": "blue", "color": "black", "weight": 1, "fillOpacity": 0.3}

mapa = folium.Map(location=[40.4168, -3.7038], zoom_start=6, control_scale=True)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    name="OpenStreetMap Español",
    attr='© OpenStreetMap contributors',
).add_to(mapa)

folium.GeoJson(
    geojson_data,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Provincia:"], sticky=True),
    highlight_function=lambda x: {"fillColor": "green", "fillOpacity": 0.7},
).add_to(mapa)

map_data = st_folium(mapa, width=800, height=600, key="map")
if map_data and "last_active_drawing" in map_data:
    last_active = map_data.get("last_active_drawing", None)
    if last_active and "properties" in last_active:
        st.session_state.selected_province = last_active["properties"].get("name", None)
        reset_prediction()

# --- Entradas de predicción ---
days_mapping = {"Lunes": "Monday", "Martes": "Tuesday", "Miércoles": "Wednesday", "Jueves": "Thursday",
                "Viernes": "Friday", "Sábado": "Saturday", "Domingo": "Sunday"}
months_mapping = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
                  "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}

input_col1, input_col2 = st.columns(2)
with input_col1:
    day_of_week_og = st.selectbox(
        "Selecciona un día de la semana",
        list(days_mapping.keys()),
        on_change=reset_prediction
    )
    day_of_week = days_mapping[day_of_week_og]
with input_col2:
    month_og = st.selectbox(
        "Selecciona un mes",
        list(months_mapping.keys()),
        on_change=reset_prediction
    )
    month = months_mapping[month_og]

# --- Botón para Calcular la Predicción ---
if st.button("Calcular predicción"):
    selected_province_original = st.session_state.selected_province
    selected_province = province_name_mapping.get(selected_province_original, selected_province_original)

    if selected_province is None:
        st.error("Por favor, selecciona una provincia en el mapa.")
    elif selected_province not in available_provinces:
        st.warning(
            f"""
            Aún no tenemos datos sobre la provincia seleccionada (**{selected_province_original}**).
            De momento, solo podemos ofrecerle datos sobre las islas:
            **Islas Baleares, Las Palmas y Santa Cruz De Tenerife**.
            Esperamos en un tiempo poder ofrecerle los datos sobre la provincia que desea.
            """
        )
    else:
        # Transformar la provincia seleccionada
        province_encoded = le_provincia_destino_name.transform([selected_province])[0]

        # Calcular predicción
        day_of_week_encoded = le_day_of_week.transform([day_of_week])[0]
        pred_data = pd.DataFrame({
            'provincia_destino_name': [province_encoded],
            'day_of_week': [day_of_week_encoded],
            'month': [month]
        })
        st.session_state.prediction = model.predict(pred_data)[0]
        st.session_state.show_prediction = True

        # --- Mostrar la predicción --- #
        if st.session_state.show_prediction and st.session_state.prediction is not None:
            st.markdown(
                f"### El próximo **{day_of_week_og}** de **{month_og}** llegarán a **{st.session_state.selected_province}** aproximadamente **{int(st.session_state.prediction)}** viajeros."
            )
        # -- -- #

        # --- Grafico datos reales vs prediccióm --- #
        u = og_data.copy()
        data_to_plot = u[(u["day_of_week"] == day_of_week) & (u["month"] == month) & (u["provincia_destino_name"] == selected_province)]

        # Visualización datos reales
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=data_to_plot, x="date", y="viajes")

        # Añadir columna de predicción
        x_pos = len(data_to_plot)
        ax.bar(x_pos, int(st.session_state.prediction), color="red")

        for i in ax.containers:
            ax.bar_label(i, padding=3)
        
        # Ajustar etiquetas del eje x
        x_labels = list(data_to_plot["date"].astype(str))
        x_labels.append("Predicción")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=40)

        plt.title(f"Número de viajes a {selected_province_original} los {day_of_week_og} de {month_og}")
        plt.tight_layout()
        st.pyplot(fig)
        # -- -- #


# --- END OF FILE --- #