import dis
from math import dist
from re import M
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# MIZU: Water Potatbility

This app predicts the **Water Potability** and classifies water based on salinity and sodium levels!
""")
st.write('---')

# Loads the Boston House Price Dataset
# boston = datasets.load_boston()
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')   

def user_input_features():
    district = st.sidebar.slider('district', 0.0, 32.0, 1.0)
    mandal = st.sidebar.slider('mandal', 0.0, 311.0, 29.0)
    village = st.sidebar.slider('village', 0.0, 364.0, 43.0)
    lat_gis = st.sidebar.slider('lat_gis', 15.896441, 19.730555, 17.666775)
    long_gis = st.sidebar.slider('long_gis', 77.444, 80.92, 80.90)
    gwl = st.sidebar.slider('gwl', 0.57, 43.17, 5.36)
    season = st.sidebar.slider('season', 0.0, 1.0, 0.0)
    pH = st.sidebar.slider('pH', 6.6, 10.44, 7.93)
    EC = st.sidebar.slider('E.C', 212.0, 5440.0, 1550.0)
    TDS = st.sidebar.slider('TDS', 135.68, 3481.6, 992.0)
    CO3 = st.sidebar.slider('C03', 0.0, 100.0, 0.0)
    HCO3 = st.sidebar.slider('HCO3', 30.0, 970.58, 547.19)
    Cl = st.sidebar.slider('Cl', 10.0, 1500.0, 140.0)
    F = st.sidebar.slider('F', 0.0, 4.97, 0.9)
    NO3 = st.sidebar.slider('NO3 ', 0.44, 735.214, 11.29)
    SO4 = st.sidebar.slider('SO4', 1.0, 453.0, 13.0)
    Na = st.sidebar.slider('Na', 5.07, 714.8, 190.99) 
    K = st.sidebar.slider('K', 0.16, 213.7, 6.0)
    Ca = st.sidebar.slider('Ca', 8.0, 488.0, 24.0)
    Mg = st.sidebar.slider('Mg', 4.86, 228.51, 72.93) 
    TH = st.sidebar.slider('T.H', 39.99, 2099.63, 359.87) 
    SAR = st.sidebar.slider('SAR', 0.20, 31.07, 4.37)
    RSC = st.sidebar.slider('RSC  meq  / L', -30.87, 17.41, 3.74)
    data = {'district': district,
            'mandal': mandal,
            'village': village,
            'lat_gis':lat_gis,
            'long_gis':long_gis, 'gwl':gwl, 'season':season,
       'pH':pH, 'E.C':EC, 'TDS':TDS, 'CO3':CO3, 'HCO3':HCO3, 'Cl':Cl, 'F':F, 'NO3 ':NO3, 'SO4':SO4, 'Na':Na, 'K':K,
       'Ca':Ca, 'Mg':Mg, 'T.H':TH, 'SAR':SAR, 'RSC  meq  / L':RSC}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Load model
with open('rf_model_binary.pkl', 'rb') as file:
    loaded_model_binary = pickle.load(file)
# Apply Model to Make Prediction
prediction = loaded_model_binary.predict(df)

st.header('Prediction of Water Potability')
st.write(prediction)
st.write('---')

# Load model
with open('rf_model_multi.pkl', 'rb') as file:
    loaded_model_multi = pickle.load(file)
# Apply Model to Make Prediction
multi_prediction = loaded_model_multi.predict(df)

st.header('Prediction of Multi-Class')
st.write(multi_prediction)
st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
