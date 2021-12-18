import base64
import streamlit as st
import catboost
import joblib
import numpy as np
import pandas as pd

@st.cache(allow_output_mutation=True)

def prediction(year,experiment,data_use,month,n_rate,pp2,pp7,air_t,daf_td,daf_sd,wfp,nh4,no3,clay,som):
    months_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                   'September': 9, 'October': 10, 'November': 11, 'December': 12}
    datause_dict = {"Building": 1, "Testing": 0}
    exp_dict = {"Arlington WI": 1, "MCSE-T2": 2, "BCSE_KBS": 3}

    exp =exp_dict[experiment]
    data = datause_dict[data_use]
    mnth =months_dict[month]

    model = joblib.load('C:/Users/Chinmay/Desktop/THMLC/N20/pkl/catboost.pkl.compressed')
    pred = model.predict([year,exp,data,mnth,n_rate,pp2,pp7,air_t,daf_td,daf_sd,wfp,nh4,no3,clay,som])

    return np.round(np.exp(pred),2)


def main():

    html_temp = """ 
        <div style ="background-color:black"> 
        <h1 style ="color:white;text-align:center;">N2O Emission Prediction</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html=True)


    col1, col2, col3, = st.columns(3)

    with col1:
        year = st.number_input('Year',min_value=2012,max_value=2018,step=1 )
        experiment = st.selectbox('Type of Experiment',('Arlington WI','MCSE-T2','BCSE_KBS'))
        data_use = st.selectbox('Select the data',('Building','Testing'))
        month = st.selectbox('Month',('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
        n_rate = st.number_input('N_rate',min_value=0)

    with col2:
        pp2 = st.number_input('PP2',min_value=0.0)
        pp7 = st.number_input('PP7',min_value=0.0)
        air_t = st.number_input('Air_temp',min_value=-20.0,max_value = 30.0 )
        daf_td = st.number_input('DAF_TD')
        daf_sd = st.number_input('DAF_sD')
    with col3:
        wfp = st.slider('WFPS25',min_value=0.0,max_value=1.0)
        nh4 = st.number_input('NH3')
        no3 = st.number_input('NO3')
        clay = st.number_input('Clay')
        som = st.number_input('SOM')


    if st.button("Predict"):
            result = prediction(year,experiment,data_use,month,n_rate,pp2,pp7,air_t,daf_td,daf_sd,wfp,nh4,no3,clay,som)
            st.success('Estimated N2O emmission is  {}'.format(result))
            print(result)
if __name__=='__main__':
    main()
