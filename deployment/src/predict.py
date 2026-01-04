import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.pkl')
    classifier = joblib.load('classifier.pkl')
    return preprocessor, classifier

def binary_mapping(X):
    """Apply binary mapping manually"""
    X = X.copy()
    X['cb_person_default_on_file'] = X['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).fillna(0)
    return X

def run():
    preprocessor, classifier = load_models()

    st.write('## Predict Default Risk')
    st.write('### Input Your Data Here')
    
    with st.form(key='Form Parameter'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', 20, 144, 26)
            income = st.number_input('Income', 0, 6000000, 55000)
            emp_length = st.number_input('Employment Length', 0, 123, 4)
            loan_amount = st.number_input('Loan Amount', 500, 35000, 8000)
            int_rate = st.number_input('Interest Rate (%)', 5.0, 25.0, 10.0, step=0.1)
            percent_income = st.number_input('DTI', 0.00, 0.85, 0.15, step=0.01, help='Debt-to-Income ratio. Input your loan percent income.')
        
        with col2:
            credit_length = st.number_input('Credit History Length', 0, 30, 4)
            home_ownership = st.selectbox('Home Ownership', ('RENT', 'MORTGAGE', 'OWN', 'OTHER'), index=0)
            loan_intent = st.selectbox('Loan Intention', 
                                      ('EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 
                                       'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'), index=2)
            loan_grade = st.selectbox('Loan Grade', ('A', 'B', 'C', 'D', 'E', 'F', 'G'), index=0)
            cb_history = st.radio('Default History', ('Y', 'N'), index=0)

        data_inf = pd.DataFrame([{
            'person_age': age,	
            'person_income': income,	
            'person_home_ownership': home_ownership,	
            'person_emp_length': emp_length,	
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amount,	
            'loan_int_rate': int_rate,	
            'loan_percent_income': percent_income,	
            'cb_person_default_on_file': cb_history,
            'cb_person_cred_hist_length': credit_length,
        }])
        
        st.dataframe(data_inf, use_container_width=True)
        submit = st.form_submit_button('Predict', type='primary', use_container_width=True)

    if submit:
        with st.spinner('Making prediction...'):
            data_processed = binary_mapping(data_inf)
            data_transformed = preprocessor.transform(data_processed)
            y_pred_inf = classifier.predict(data_transformed)
            y_pred_proba = classifier.predict_proba(data_transformed)
        
        st.divider()
        
        if y_pred_inf[0] == 1:
            st.error('### High Risk: Default Predicted')
            st.image('denied.png')
            st.write(f"**Probability of Default:** {y_pred_proba[0][1]:.2%}")
        else:
            st.success('### Low Risk: Non-Default Predicted')
            st.image('approved.png')
            st.write(f"**Probability of Non-Default:** {y_pred_proba[0][0]:.2%}")
        
        with st.expander("View Detailed Probabilities"):
            prob_df = pd.DataFrame({
                'Outcome': ['Non-Default (0)', 'Default (1)'],
                'Probability': [f"{y_pred_proba[0][0]:.2%}", f"{y_pred_proba[0][1]:.2%}"]
            })
            st.dataframe(prob_df, use_container_width=True)