import dis
from math import dist
from re import M
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# MIZU: Water Potatbility

This app predicts the **Water Potability** and classifies water based on salinity and sodium levels!
""")
st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')   

district = 0
mandal = 0
village = 0
lat_gis = 0
long_gis = 0
season = 0
CO3 = 0
HCO3 = 0
Cl = 0
F = 0
NO3 = 0
SO4 = 0
Na = 0
K = 0
Ca = 0
Mg = 0

def user_input_features():
    district = st.sidebar.slider('district', 0.0, 32.0, 1.0)
    mandal = st.sidebar.slider('mandal', 0.0, 311.0, 29.0)
    village = st.sidebar.slider('village', 0.0, 364.0, 43.0)
    lat_gis = st.sidebar.slider('lat_gis', 15.896441, 19.730555, 17.666775)
    long_gis = st.sidebar.slider('long_gis', 77.444, 80.92, 80.90)
    season = st.sidebar.slider('season', 0.0, 1.0, 0.0)
    CO3 = st.sidebar.slider('C03', 0.0, 100.0, 40.0)
    HCO3 = st.sidebar.slider('HCO3', 30.0, 970.58, 171.86)
    Cl = st.sidebar.slider('Cl', 10.0, 1500.0, 120.0)
    F = st.sidebar.slider('F', 0.0, 4.97, 0.83)
    NO3 = st.sidebar.slider('NO3 ', 0.44, 735.214, 86.35)
    SO4 = st.sidebar.slider('SO4', 1.0, 453.0, 14.0)
    Na = st.sidebar.slider('Na', 5.07, 714.8, 84.85) 
    K = st.sidebar.slider('K', 0.16, 213.7, 8.0)
    Ca = st.sidebar.slider('Ca', 8.0, 488.0, 104.0)
    Mg = st.sidebar.slider('Mg', 4.86, 228.51, 4.862) 
    data = {
        # 'district': district,
            # 'mandal': mandal,
            # 'village': village,
            # 'lat_gis':lat_gis,
            # 'long_gis':long_gis, 'gwl':gwl, 'season':season,
    #    'pH':pH, 'E.C':EC, 'TDS':TDS, 
       'CO3':CO3, 'HCO3':HCO3, 'Cl':Cl, 'F':F, 'NO3 ':NO3, 'SO4':SO4, 'Na':Na, 'K':K,
       'Ca':Ca, 'Mg':Mg, 
    #    'T.H':TH, 'SAR':SAR, 'RSC  meq  / L':RSC
            }
    features = pd.DataFrame(data, index=[0])
    return features

df_imputation = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df_imputation)
st.write('---')

# Load model
with open('stack_model.pkl', 'rb') as file:
    loaded_model_stack = pickle.load(file)
# Apply Model to Make Prediction

EC = loaded_model_stack[0].predict(df_imputation)
TDS = loaded_model_stack[1].predict(df_imputation)
pH = loaded_model_stack[2].predict(df_imputation)
TH = loaded_model_stack[3].predict(df_imputation)
SAR = loaded_model_stack[4].predict(df_imputation)
RSC = loaded_model_stack[5].predict(df_imputation)
gwl = loaded_model_stack[6].predict(df_imputation)

op_imputation = pd.DataFrame({'EC':EC, 'TDS':TDS, 'pH':pH, 'TH':TH, 'SAR':SAR, 'RSC':RSC, 'gwl':gwl})

st.header('Imputation of Water from Minerals')
st.write(op_imputation)
st.write('---')

def final_input_features():
    
    data = {'district': district,
            'mandal': mandal,
            'village': village,
            'lat_gis':lat_gis,
            'long_gis':long_gis, 'gwl':gwl, 'season':season,
       'pH':pH, 'E.C':EC, 'TDS':TDS, 'CO3':CO3, 'HCO3':HCO3, 'Cl':Cl, 'F':F, 'NO3 ':NO3, 'SO4':SO4, 'Na':Na, 'K':K,
       'Ca':Ca, 'Mg':Mg, 'T.H':TH, 'SAR':SAR, 'RSC  meq  / L':RSC}
    features = pd.DataFrame(data, index=[0])
    return features

df = final_input_features()

# # Load model
with open('rf_model_binary.pkl', 'rb') as file:
    loaded_model_binary = pickle.load(file)
# # Apply Model to Make Prediction
prediction = loaded_model_binary.predict(df)

st.header('Prediction of Water Potability')
if prediction==0:
    st.write('**Drinkable**')
else:
    st.write('**Non-drinkable**')
st.write('---')

# # Load model
with open('rf_model_multi.pkl', 'rb') as file:
    loaded_model_multi = pickle.load(file)
# # Apply Model to Make Prediction
multi_prediction = loaded_model_multi.predict(df)

st.header('Prediction of Multi-Class')
if multi_prediction == 3:
    st.write('**C3S1**')
elif multi_prediction == 1:
    st.write('**C2S1**')
elif multi_prediction == 7:
    st.write('**C4S1**')
elif multi_prediction == 8:
    st.write('**C4S2**')
elif multi_prediction == 10:
    st.write('**C4S4**')
elif multi_prediction == 4:
    st.write('**C3S2**')
elif multi_prediction == 0:
    st.write('**C1S1**')
elif multi_prediction == 5:
    st.write('**C3S3**')
elif multi_prediction == 9:
    st.write('**C4S3**')
elif multi_prediction == 6:
    st.write('**C3S4**')
elif multi_prediction == 2:
    st.write('**C2S2**')
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
