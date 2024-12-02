# DataNomads - Spanish Mobility Analysis

## Overview
DataNomads analyzes and predicts travel patterns across Spanish provinces using machine learning techniques. The project focuses on understanding mobility between spanish mainland and its islands.

## Features
- General mobility analysis
- Prediction of daily provincial visitor inflow
- Interactive visualization dashboard

## Tech Stack
- Python 3.7+
- Streamlit
- Folium
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository
```sh
git clone https://github.com/pdpau/DataNomads.git
cd DataNomads
```

2. Install dependencies
```sh
pip install -r requirements.txt
```

3. Run the Streamlit app
```sh
cd code
cd web
streamlit run main.py
```

4. Open the browser and go to http://localhost:8501

## Project Structure
```
code/
├── web/
│   ├── main.py
│   ├── dataset.csv
│   ├── model.pkl
│   ├── le_day_of_week.pkl
│   ├── le_provincia_destino_name.pkl
├── data/
├── main_notebook.ipynb
├── exploratory_data_analysis.ipynb
├── requirements.txt
├── README.md
```

## Team
- Pau Peirats
- Oriol Bech
- Fernando Sánchez-Mora
- Román Bermejo
- Martí Lluch
