import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title='Credit Risk Loan - Exploratory Data Analysis',
    layout='centered',
    initial_sidebar_state='expanded'
)

def run():
    st.title('Exploratory Data Analysis on Credit Risk Data')
    st.image('analyst.jpg')

    st.markdown('---')
    
    df = pd.read_csv('credit_risk_dataset.csv')
    st.dataframe(df)

    st.write('### 1. Distribution of Numerical Column')
    st.write('> Analyzing the general distribution on numeric columns ')

    numerical_columns = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ]

    selected_num_col = st.selectbox(
        "Select a numerical column to visualize distribution:",
        numerical_columns
    )

    units = {
    'person_age': 'years',
    'person_income': '$',
    'person_emp_length': 'years',
    'loan_amnt': '$',
    'loan_int_rate': '%',
    'loan_percent_income': '%',
    'cb_person_cred_hist_length': 'years'
    }

    if selected_num_col:
        unit = units.get(selected_num_col, '')
        xlabel = f"{selected_num_col} ({unit})" if unit else selected_num_col
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.suptitle(f'Distribution of {selected_num_col}', y=0.96)
        sns.histplot(df[selected_num_col], kde=True, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.4)
        st.pyplot(fig)

    st.markdown('---')

    st.write('### 2. Distribution of Loan Status')
    st.write('> Analyzing the general distribution of Loan Status')

    warna = sns.color_palette('tab10', 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        df['loan_status'].value_counts().sort_index(),
        labels=['Non-Default', 'Default'],
        colors=warna,
        autopct='%1.1f%%'
    )
    ax.set_title('Distribustion of Loan Status')
    st.pyplot(fig)

    st.write('> Distribution of Loan Status, majority Non-Default for 78.2%, and for Default 21.8%')

    st.markdown('---')

    st.write('### 3. How Percent Income Loan Affects Loan Status')

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x='loan_status',
        y='loan_percent_income',
        hue='loan_status',
        palette='tab10',
        ax=ax
    )
    ax.set_title("Loan Percent Income vs Loan Status")
    ax.set_xlabel("Loan Status (0=Non-Default, 1=Default)")
    ax.set_ylabel("Loan Percent Income")
    st.pyplot(fig)

    st.write('>Lower `loan_percent_income` has a lower risk to be Default, vice versa. Outliers explain that there are some factor that affect higher `loan_percent_income` to be Non-Default')

    st.write('### 4. Distribution of Categorical Column')

    categorical_columns = ['person_home_ownership', 'loan_intent',
                        'loan_grade', 'cb_person_default_on_file']

    selected_cat_col = st.selectbox(
        "Select a numerical column to visualize distribution:",
        categorical_columns
    )

    if selected_cat_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.suptitle(f'Distribution of {selected_cat_col}', y=0.96)
        sns.countplot(data=df,
                      x=selected_cat_col,
                      hue=selected_cat_col,
                      ax=ax,
                      palette='tab10'
        )
        ax.set_xlabel(selected_cat_col)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig)

if __name__ == '__main__':
    run()