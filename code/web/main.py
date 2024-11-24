# Imports
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import folium
from streamlit_folium import st_folium
import base64
import os

# Page config
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

# --- Resto del código ---
# Aquí continúa tu lógica de predicción y visualización



# --- Cargar y procesar datos --- #
data = pd.read_csv("model_data.csv")
data.drop(columns=['date', 'year'], inplace=True)

le_day_of_week = LabelEncoder()
le_provincia_destino_name = LabelEncoder()
data['day_of_week'] = le_day_of_week.fit_transform(data['day_of_week'])
data['provincia_destino_name'] = le_provincia_destino_name.fit_transform(data['provincia_destino_name'])

# Dividir datos
X = data.drop(columns=['viajes'])
y = data['viajes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Modelo de predicción
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# --- Inicializar session_state --- #
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "province_original" not in st.session_state:
    st.session_state.province_original = None

# --- Página principal --- #
#st.markdown("# DataNomads")
st.markdown("#### Bienvenido a la aplicación de predicción de viajes nacionales")

st.divider()

# --- Entradas de predicción --- #
days_mapping = {"Lunes": "Monday", "Martes": "Tuesday", "Miércoles": "Wednesday", "Jueves": "Thursday",
                "Viernes": "Friday", "Sábado": "Saturday", "Domingo": "Sunday"}
months_mapping = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
                  "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}
province_mapping = {"Islas Baleares": "Balears, Illes", "Las Palmas": "Palmas, Las", "Santa Cruz de Tenerife": "Santa Cruz de Tenerife"}

input_col1, input_col2, input_col3 = st.columns(3)
with input_col1:
    day_of_week_og = st.selectbox("Selecciona un día de la semana", list(days_mapping.keys()))
    day_of_week = days_mapping[day_of_week_og]
with input_col2:
    month_og = st.selectbox("Selecciona un mes", list(months_mapping.keys()))
    month = months_mapping[month_og]
with input_col3:
    province_og = st.selectbox("Selecciona una provincia de destino", list(province_mapping.keys()))
    province = province_mapping[province_og]

# --- Botón para calcular la predicción --- #
if st.button("Calcular predicción"):
    province_encoded = le_provincia_destino_name.transform([province])[0]
    day_of_week_encoded = le_day_of_week.transform([day_of_week])[0]
    pred_data = pd.DataFrame({
        'provincia_destino_name': [province_encoded],
        'day_of_week': [day_of_week_encoded],
        'month': [month]
    })
    st.session_state.prediction = gbr.predict(pred_data)[0]
    st.session_state.province_original = le_provincia_destino_name.inverse_transform([province_encoded])[0]

# Mostrar la predicción si está calculada
if st.session_state.prediction is not None:
    st.markdown(f"### Predicción: {int(st.session_state.prediction)} viajeros para un *{day_of_week_og}* de *{month_og}* en *{province_og}*.")

# --- Mapa interactivo --- #
if st.session_state.province_original:
    province_coords = {
        "Balears, Illes": [39.57119, 2.64663],
        "Palmas, Las": [28.123546, -15.436257],
        "Santa Cruz de Tenerife": [28.46363, -16.251847]
    }

    st.markdown("## Mapa de España con la provincia marcada")
    mapa = folium.Map(location=[40, -3], zoom_start=6)
    if st.session_state.province_original in province_coords:
        folium.Marker(location=province_coords[st.session_state.province_original], tooltip=st.session_state.province_original).add_to(mapa)
        st_folium(mapa, width=700, height=500)
    else:
        st.error("No se encontraron coordenadas para la provincia seleccionada.")

