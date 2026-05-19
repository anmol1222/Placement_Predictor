import pandas as pd
import numpy as np
import joblib,pickle

#load the model
model=pickle.load(open('placement.pkl','rb'))
accu_sco=joblib.load("accuracy.pkl",'rb')

placed_val={0:'NOT Placed',1:'Placed'}
#now add the progress

import streamlit as st
st.title(':rainbow[The Placement Prediction]')

import time
prg=st.progress(0)

for i in range(100):
    time.sleep(0.002)
    prg.progress(i+1)
    
 
cgpa=st.number_input("Enter the CGPA",1,10) 
resume_sco=st.number_input("Enter the Resume score",1,10) 

if st.button("Predict"):
    with st.spinner("Predicting....."):
        features=np.array([[cgpa,resume_sco]])
        prediction=model.predict(features)[0]
        result_text=placed_val[prediction]
        # accu_sco=model.score(accu_sco)

    st.success(f"Prediction of Placement: {result_text}")    
    st.write(f"Model Accuracy: {accu_sco *100:.2f} %")
    
    if prediction==1:
        st.balloons()
    else:
        st.snow()
            

    