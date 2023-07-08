import streamlit as st 
import pickle 
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

rfs = pickle.load(open("randomforest.sav", "rb")) 
xgb = pickle.load(open("xgboost.sav", "rb"))




st.title("Alzheimer predictions using ML")

introduction =pd.DataFrame({
"Title" :["Sex","Age",'educ','ses','cdr','mmse','etiv','nwbv','asf'],
"Description" : ["Age",'Sex','Years of education','Socioeconomic status','Clinical dimentia rating','Mini mental state examination','Estimated total intracranical volumne','Normalized whole brain volume','Atlas scaling factor']
})
st.table(introduction)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu("Health Assistant",
                            ["Random Forest Model",
                            "XGBoost Model",],
                            default_index=0)


col1, col2, col3 = st.columns(3)

with col1:
    sex = st.text_input('Sex (M/F)')
    if sex == 'M':
       sex = 1
    else:
       sex = 0
    
with col1:
    age = st.text_input('Age')

with col1:
    educ = st.text_input('EDUC')
    # educ = float(educ)

with col2:
    ses = st.text_input('SES')



with col2:
    mmse = st.text_input('MMSE')

with col2:
    cdr = st.text_input('CDR')

with col3:
    etiv = st.text_input('ETIV')

with col3:
    nwbv = st.text_input('NWBV')

with col3:
    asf = st.text_input("ASF")




if st.button('Alzheimer Test Result'):
    inputs = [sex,age,educ,ses,mmse,cdr,etiv,nwbv,asf]
    print(inputs)
    inputs = [float(i) for i in inputs]
    print(inputs)
    # inputs = [1.0,87.0,14.0,2.0,27.0,0.0,1987.0,0.696,0.883]
    print(inputs)
    features = np.asarray(inputs).reshape(1,-1)
    if (selected == "Random Forest Model"):
        alz_prediction = rfs.predict(features)
    if (selected == "XGBoost Model"):
        alz_prediction = xgb.predict(features)

    print(alz_prediction) 
    
    
    # st.write(alz_prediction)
    # print(alz_prediction)
    
    
    if (alz_prediction[0] == 0):
        alz_diagnosis = 'The patient is Converted'
        st.warning(alz_diagnosis)
    elif (alz_prediction[0] == 1):
        alz_diagnosis = 'The patient is Demented'
        st.error(alz_diagnosis)
    elif (alz_prediction[0] == 2):
        alz_diagnosis = 'The patient is Non demented'
        st.success(alz_diagnosis)



# if (selected == "XGBoost Model"):
    
#     # code for Prediction
#     alz_diagnosis = ''
    
#     # creating a button for Prediction
    
#     if st.button('Alzheimer Test Result'):
#         alz_prediction = xgb.predict([[sex,age,educ,ses,mmse,cdr,etiv,nwbv,asf]])
        
#         if (alz_prediction[0] == 1):
#           alz_diagnosis = 'The person is prone to Alzheimer'
#           st.warning(alz_diagnosis)
#         else:
#           alz_diagnosis = 'The person is not prone to Alzheimer'
#           st.success(alz_diagnosis)


    
    
    

