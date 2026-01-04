import streamlit as st
import eda
import predict

st.title('Credit Risk Prediction')

tab1, tab2, tab3 = st.tabs(['Home', 'Predict Default Risk', 'EDA'])

with tab1:
    st.title('WELCOME')
    st.image('Na_Nov_04.jpg')
    st.write('')
    st.write('You may use this model to predict if individual might default or not')
    st.write("Go to 'Predict Default Risk' tab to use the prediction model")
    st.write("Go to 'EDA' tab to see the analysis of the data")
    st.write('Machine Learning model is based on Decision Tree')
    

with tab2:
    predict.run()

with tab3:
    eda.run()