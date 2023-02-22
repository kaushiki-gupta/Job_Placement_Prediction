import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle

df = pickle.load(open('df1.pkl','rb'))
model = pickle.load(open('lr1.pkl','rb'))

st.title('Job Placement Prediction')
st.header('Fill the details to predict the Placement Status')

# Features

# numerical columns
ssc_percentage = st.number_input('SSC Percentage')
hsc_percentage = st.number_input('HSC Percentage')
degree_percentage = st.number_input('Degree Percentage')
emp_test_percentage = st.number_input('EMP Test Percentage')
mba_percent = st.number_input('MBA Percent')

# categorical columns
gender = st.selectbox('Gender',df['gender'].unique())
ssc_board = st.selectbox('SSC Board',df['ssc_board'].unique())
hsc_board = st.selectbox('HSC Board',df['hsc_board'].unique())
hsc_subject = st.selectbox('HSC Subject',df['hsc_subject'].unique())
undergrad_degree = st.selectbox('Undergrad Degree',df['undergrad_degree'].unique())
work_experience = st.selectbox('Work Experience',df['work_experience'].unique())
specialisation = st.selectbox('Specialisation',df['specialisation'].unique())

if st.button('Predict Placemeent Status'):
    test_data = np.array([gender, ssc_percentage, ssc_board, hsc_percentage, hsc_board,
       hsc_subject, degree_percentage, undergrad_degree,
       work_experience, emp_test_percentage, specialisation,mba_percent])
    test_data = test_data.reshape([1,12])
    st.success(model.predict(test_data)[0])

# to start => streamlit run ml_app1.py
# to stop => control + C