# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 20:22:33 2022

@author: Acer
"""

import streamlit as st
import os
import numpy as np
import pickle

MODEL_PATH = os.path.join(os.getcwd(),'best_model_breast.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)


#%% STREAMLIT
with st.form("Patient's Form"):
    st.title("Type of Breast Cancer Prediction")
    st.video("https://youtube.com/watch?v=FTH56ifHt28&si=EnSIkaIECMiOmarE", format="video/mp4") 
    # credit video: "Common Types of Breast Cancer" By Mayo Clinic YouTube channel
    st.header("Let's check the type of your breast cancer!")
    radius_mean = int(st.number_input("Key in your radius of lobes: "))
    texture_mean = int(st.number_input("Key in your mean of surface texture: "))
    perimeter_mean = int(st.number_input("Key in your outer perimeter of lobes: "))
    area_mean = int(st.number_input("Key in your mean area of lobes: "))
    concavity_mean = int(st.number_input("Key in the mean of concavity: "))
    radius_se = int(st.number_input("Key in the radius of se: "))
    perimeter_se = int(st.number_input("Key in the perimeter of se: "))
    area_se = int(st.number_input("Key in the area of se: "))
    radius_worst = int(st.number_input("Key in your breast worst radius: "))
    texture_worst = int(st.number_input("Key in your breast worst texture: "))
    perimeter_worst = int(st.number_input("Key in your breast worst perimeter: "))
    area_worst = int(st.number_input("Key in your breast worst area: "))
    compactness_worst = int(st.number_input("Key in your breast worst compactness: "))
    concavity_worst = int(st.number_input("Key in your breast worst concavity: "))
    concave_points_worst = int(st.number_input("Key in your breast worst concave points: "))
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Radius of lobes:",radius_mean,
                 "Mean of surface texture:",texture_mean,
                 "Outer perimeter of lobes:",perimeter_mean,
                 "Mean area of lobes:",area_mean,
                 "Mean of concavity:",concavity_mean,
                 "Radius of se:",radius_se,
                 "Perimeter of se:",perimeter_se,
                 "Area of se:",area_se,
                 "Worst radius:",radius_worst,
                 "Worst texture:",texture_worst,
                 "Worst perimeter:",perimeter_worst,
                 "Worst area:",area_worst,
                 "Worst compactness:",compactness_worst,
                 "Worst concavity:",concavity_worst,
                 "Worst concave points:",concave_points_worst)
        temp = np.expand_dims([radius_mean,texture_mean,perimeter_mean,area_mean,concavity_mean,
                               radius_se,perimeter_se,area_se,radius_worst,texture_worst,
                               perimeter_worst,area_worst,compactness_worst,concavity_worst,
                               concave_points_worst], axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Breast cancer type benign',
                        1:'Breast cancer type malignant'}
        
        if outcome == 1:
            st.snow()
            st.markdown('Your breast cancer type is malignant!')
            st.write("Please consult with you doctor for further medical assessment!")
            st.image("https://cdn.cordlife.sg/images/infographics/india/malignant-tumor-types.jpg")
            # Credit pic: cdn.cordlife.sg website
        else:
            st.balloons()
            st.markdown('Your breast cancer type is benign!')
            st.write("Your breast cancer is not as fatal as malignant tumor since the tumor will not spread out!")
            st.write("Please consult with you doctor for further medical assessment!")
            st.image("https://www.lifebiotic.com/wp-content/uploads/breast_cancer_infographic_lifebiotic.png")
            # Credit pic: lifebiotics.com website
        
