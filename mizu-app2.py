import dis
from math import dist
from re import M
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.write("""
# MIZU: Forecasting Water Quality Factors and Predicting Drinkability

This app predicts the *Water Potability* and classifies water based on salinity and sodium levels through its unique multi-input-output nature by using minerals as input parameters!
""")
st.write('---')

 
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
    st.header('Location details') 
    image = open('Mizu_telanagan.jpg', 'rb').read()
    st.image(image)
    # st.write("""⇒Fig : Mandal wise rainfall deviation for water year 2021-22""")
    district = st.text_input('District', '0')
    try:
        district = float(district)
    except ValueError:
        district = None

    mandal = st.text_input('Mandal', '0')
    try:
        mandal = float(mandal)
    except ValueError:
        mandal = None

    village = st.text_input('Village', '0')
    try:
        village = float(village)
    except ValueError:
        village = None

    lat_gis = st.text_input('Lat_gis', '0')
    try:
        lat_gis = float(lat_gis)
    except ValueError:
        lat_gis = None

    long_gis = st.text_input('Long_gis', '0')
    try:
        long_gis = float(long_gis)
    except ValueError:
        long_gis = None

    season = st.text_input('Season', '0')
    try:
        season = float(season)
    except ValueError:
        season = None
    st.write('---')

    st.header('Specify Mineral Parameters') 
    CO3 = st.text_input('C03', '0')
    try:
        CO3 = float(CO3)
    except ValueError:
        CO3 = None

    HCO3 = st.text_input('HCO3', '0')
    try:
        HCO3 = float(HCO3)
    except ValueError:
        HCO3 = None

    Cl = st.text_input('Cl', '0')
    try:
        Cl = float(Cl)
    except ValueError:
        Cl = None

    F = st.text_input('F', '0')
    try:
        F = float(F)
    except ValueError:
        F = None

    NO3 = st.text_input('NO3 ', '0')
    try:
        NO3 = float(NO3)
    except ValueError:
        NO3 = None

    SO4 = st.text_input('SO4', '0')
    try:
        SO4 = float(SO4)
    except ValueError:
        SO4 = None

    Na = st.text_input('Na', '0')
    try:
        Na = float(Na)
    except ValueError:
        Na = None

    K = st.text_input('K', '0')
    try:
        K = float(K)
    except ValueError:
        K = None

    Ca = st.text_input('Ca', '0')
    try:
        Ca = float(Ca)
    except ValueError:
        Ca = None

    Mg = st.text_input('Mg', '0')
    try:
        Mg = float(Mg)
    except ValueError:
        Mg = None
    st.write('---')
    data = {
       'CO3':CO3, 'HCO3':HCO3, 'Cl':Cl, 'F':F, 'NO3 ':NO3, 'SO4':SO4, 'Na':Na, 'K':K,
       'Ca':Ca, 'Mg':Mg
       }

    features = pd.DataFrame(data, index=[0])

    return features

df_imputation = user_input_features()

st.header('Given Input Parameters to the Model')
st.write(df_imputation)
st.write('---')

with open('stack_model.pkl', 'rb') as file:
    loaded_model_stack = pickle.load(file)

EC = loaded_model_stack[0].predict(df_imputation)
TDS = loaded_model_stack[1].predict(df_imputation)
pH = loaded_model_stack[2].predict(df_imputation)
TH = loaded_model_stack[3].predict(df_imputation)
SAR = loaded_model_stack[4].predict(df_imputation)
RSC = loaded_model_stack[5].predict(df_imputation)
gwl = loaded_model_stack[6].predict(df_imputation)

op_imputation = pd.DataFrame({'EC':EC, 'TDS':TDS, 'pH':pH, 'TH':TH, 'SAR':SAR, 'RSC':RSC, 'gwl':gwl})



st.header('Imputation of Water from Minerals (Multi-Input-Output)')
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

with open('rf_model_binary.pkl', 'rb') as file:
    loaded_model_binary = pickle.load(file)
prediction = loaded_model_binary.predict(df)

st.header('Prediction of Water Potability')
if prediction==0:
    st.write('*Drinkable*')
else:
    st.write('*Non-drinkable*')
st.write('---')

with open('rf_model_multi.pkl', 'rb') as file:
    loaded_model_multi = pickle.load(file)
multi_prediction = loaded_model_multi.predict(df)

st.header('Prediction of Multi-Class')
if multi_prediction == 3:
    st.write('**C3S1 : The low sodium and high salinity waters necessitate adequate drainage. It is important to choose crops that can tolerate salt.**')
elif multi_prediction == 1:
    st.write('**C2S1 : Waters with a medium salinity and low sodium content are suitable for irrigation and can be applied to practically all soil types with little risk of raising exchangeable sodium levels to dangerous levels even if a substantial amount of leaching takes place. Without taking any specific measures to manage salinity, crops can be cultivated.**')
elif multi_prediction == 7:
    st.write('**C4S1 : Waters with very low sodium content and excessive salinity are not appropriate for irrigation unless the soil is porous and has sufficient drainage. To achieve significant leaching, excessive irrigation water must be used. Crops that can withstand salt must be chosen.**')
elif multi_prediction == 8:
    st.write('*C4S2 : Medium salt and very high salinity fluids should not be used for irrigationon fine-textured soils with low leaching conditions, but they can be utilized on organic or coarse-textured soils with good permeability*')
elif multi_prediction == 10:
    st.write('**C4S4 : Waters with extremely high salt and salinity levels are typically unsuitablefor irrigation. These waters can present sodium concerns since they contain sodium chloride. It can be applied on coarse-textured soils with excellent drainage for crops that can withstand a lot of salt. The usage of these waters is made possible by gypsum modifications.**')
elif multi_prediction == 4:
    st.write('*C3S2 : The coarse-textured or organic soils with good permeability can be employed with the high salinity and medium sodium waters, which need good drainage*')
elif multi_prediction == 0:
    st.write('*C1S1 : Low salinity and low sodium waters are suitable for irrigation and can be utilized with the majority of crops on most soils without restriction*')
elif multi_prediction == 5:
    st.write('**C3S3 : These high sodium and salinity waters necessitate particular soil management, good drainage, significant leaching, and additions of organic matter. The usage of these waters is made possible by gypsum modifications.**')
elif multi_prediction == 9:
    st.write('*C4S3 : The majority of soils are adversely affected by dangerous amounts of exchangeable sodium produced by very high salinity and high sodium waters, which calls for particular soil management, good drainage, high leaching, and organic matter additions. The usage of these waters is made possible by the Gypsum amendment*')
elif multi_prediction == 6:
    st.write('*C3S4 : These high salinity and very high sodium waters can pose significant challenges for various purposes, including drinking, agriculture, and industry*')
elif multi_prediction == 2:
    st.write('**C2S2 : Medium salinity and medium sodium waters typically fall into the category of brackish water. Brackish water is characterized by moderate salinity levels and a moderate concentration of dissolved sodium ions.**')
st.write('---')
st.header('MIZU Framework')
image1 = open('Mizu_arch.jpg', 'rb').read()
st.image(image1)
st.write('---')
