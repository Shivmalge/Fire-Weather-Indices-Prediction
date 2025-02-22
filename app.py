import streamlit as st
import pandas as pd
import pickle
import numpy as np


st.header('Temperature FWI(Fire Weather Indices) Recommender')

inputs = []

labels = ["Temperature", "RH (Relative Humidity)", "Ws (Wind Speed)", "Rain", "FFMC (Fine Fuel Moisture Code)", "DMC (Duff Moisture Code)", "ISI (Initial Spread Index)", "Classes (Not Fire, Fire)", "Region"]

for label in labels:
    if label in ["Classes (Not Fire, Fire)", "Region"]:  # For "Classes" and "Region", use a dropdown
        input_value = st.selectbox(label, options=[0,1], key=label)
        inputs.append(input_value)
    else:
        input_value = st.text_input(label)
        try:
            inputs.append(float(input_value) if input_value else np.nan)
        except ValueError:
            inputs.append(np.nan)
    

ridge_model = pickle.load(open('ridge.pkl','rb'))
standard_scaler = pickle.load(open('scaler.pkl','rb'))

def predict(inputs):
    new_data_scaled = standard_scaler.transform([inputs])
    result = ridge_model.predict(new_data_scaled)
    return result[0]

if(st.button("Submit")):
    if any(np.isnan(inputs)):  # Check if any input is missing
        st.error("Please fill all fields with valid numbers.")
    else:
        st.write("You entered the following values:")
        for i, value in enumerate(inputs):
            st.write(f"{labels[i]}: {value}")
        
        prediction = predict(inputs)

        st.write("Predicted FWI value is :",prediction)



