# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# https://github.com/Kanaries/pygwalker

# Page config
st.set_page_config(
    page_title="DataNomads",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA



# MAIN PAGE
st.title("DataNomads")
st.write("Welcome to the DataNomads web app. This app is designed to help you visualize and analyze data. Use the sidebar to navigate through the app.")

# --- INPUTS of the prediction --- #https://docs.streamlit.io/develop/api-reference/widgets
#Creamos 3 columnas para alinear los inputs horizontalmente
input_col1, input_col2, input_col3 = st.columns(3)
with input_col1:
    day_of_week = st.selectbox("Select a day of the week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
with input_col2:
    month = st.selectbox("Select a month",
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    )
with input_col3:
    province = st.selectbox("Select a province",
        ["Balears, Illes", "Palmas, Las", "Santa Cruz de Tenerife"]
    )
# --- End of INPUTS ---


# --- OUTPUTS of the prediction ---
st.subheader("La predicción de VIAJES para ${day_of_week} en ${month} en ${province} es de: ")
st.write("Output numero de personas: ...")
st.write("Output : ...")
# --- End of OUTPUTS ---


# --- MAPA de España con la provincia marcada ---
st.write("Mapa de España con la provincia marcada")

map_df = pd.DataFrame({
    'lat': [40.463667, 39.399872, 28.291564],
    'lon': [-3.74922, -4.119863, -16.629130],
    'name': ['Madrid', 'Barcelona', 'Canarias']
}) #mapa de prueba
st.map(map_df)
# --- End of MAPA ---