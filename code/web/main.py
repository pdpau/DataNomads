# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# https://github.com/Kanaries/pygwalker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#import xgboost as xgb
#from xgboost import XGBRegressor


# Page config
st.set_page_config(
    page_title="DataNomads",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA
data = pd.read_csv("model_data.csv")
#st.write(data)

# CLEAN DATA
data.drop(columns=['date'], inplace=True)
data.drop(columns=['year'], inplace=True)

# LABEL ENCODING
le_day_of_week = LabelEncoder()
le_provincia_destino_name = LabelEncoder()
data['day_of_week'] = le_day_of_week.fit_transform(data['day_of_week'])
data['provincia_destino_name'] = le_provincia_destino_name.fit_transform(data['provincia_destino_name'])

# MODEL (Aquest es nomes una prova, es fara quan tinguem el millor model i parametres)
X = data.drop(columns=['viajes'])
y = data['viajes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
#y_pred = gbr.predict(X_test)


# --- MAIN PAGE --- #
st.markdown("# DataNomads")
st.markdown("#### Bienvenido a la aplicación de predicción...")

st.divider()

# --- INPUTS of the prediction --- #https://docs.streamlit.io/develop/api-reference/widgets
# Diccionarios para mapear días, meses y provincias
days_mapping = { "Lunes": "Monday", "Martes": "Tuesday", "Miércoles": "Wednesday", "Jueves": "Thursday", "Viernes": "Friday", "Sábado": "Saturday", "Domingo": "Sunday" }
months_mapping = { "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12 }
province_mapping = { "Islas Baleares": "Balears, Illes", "Las Palmas": "Palmas, Las", "Santa Cruz de Tenerife": "Santa Cruz de Tenerife" }

# Creamos 3 columnas para alinear los inputs horizontalmente
input_col1, input_col2, input_col3 = st.columns(3)
with input_col1:
    day_of_week_og = st.selectbox("Selecciona un día de la semana",
        ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    )
    day_of_week = days_mapping[day_of_week_og]
with input_col2:
    month_og = st.selectbox("Selecciona un mes",
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    )
    month = months_mapping[month_og]
with input_col3:
    province_og = st.selectbox("Selecciona una provincia de destino",
        ["Islas Baleares", "Las Palmas", "Santa Cruz de Tenerife"]
    )
    province = province_mapping[province_og]
# --- End of INPUTS ---

# --- OUTPUTS of the prediction ---
st.markdown(f"### La predicción de VIAJES NACIONALES para un *{day_of_week_og}* de *{month_og}* en *{province_og}* es de: ")
if st.button("Calcular predicción"):
    province = le_provincia_destino_name.transform([province])[0]
    day_of_week = le_day_of_week.transform([day_of_week])[0]
    pred_data = pd.DataFrame({
        'provincia_destino_name': [province],
        'day_of_week': [day_of_week],
        'month': [month]
    })
    prediction = gbr.predict(pred_data)
    st.write(f"Predicción: {int(prediction[0])} viajeros")
# --- End of OUTPUTS ---

st.divider()

# --- MAPA de España con la provincia marcada ---
st.markdown("## Mapa de España con la provincia marcada")

map_df = pd.DataFrame({
    'lat': [40.463667, 39.399872, 28.291564],
    'lon': [-3.74922, -4.119863, -16.629130],
    'name': ['Madrid', 'Barcelona', 'Canarias']
}) #mapa de prueba
st.map(map_df)
# --- End of MAPA ---