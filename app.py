import streamlit as st
from helper import FeatureScaling, read_data, update_secondary_col, num_columns, HandlingMissingValues, MathamaticalTRansformation
from helper import cat_columns

st.set_page_config(
     page_title="FeatureLab - Choose the best feature for your data",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/everydaycodings',
         'About': "github code link: https://github.com/everydaycodings/FeatureLab"
     }
)

st.header("Welcome to FeatureLab")

upload_file = st.file_uploader("Upload your data: ")
engineering_choice = st.selectbox("Select your Feature Engineering Method: ", options=["Feature Scaling", "Handling Missing Values", "Mathamatical Transformation"])

if upload_file is not None:

    data = read_data(upload_file)

    st.subheader(" ")
    if engineering_choice == "Feature Scaling":
        st.subheader("You Selected Feature Scaling")
        choice = st.multiselect("Select Your Prefered Choice: ", options=["Standardization", "Normalization"])
        
        if "Standardization" in choice:

            st.subheader("Standardization Scaling")
            column1 = st.selectbox("Select Your First Column: ", options=num_columns(data))
            column2 = st.selectbox("Select Your Second Column: ", options=update_secondary_col(num_columns(data), column1))
            st.subheader("Effects of Scaling")
            FeatureScaling(data).steandardization(column1, column2)
        
        if "Normalization" in choice:
            st.subheader("Normalization(MinMaxScaling) Scaling")
            column1 = st.selectbox("Select Your First Column: ", options=num_columns(data))
            column2 = st.selectbox("Select Your Second Column: ", options=update_secondary_col(num_columns(data), column1))
            st.subheader("Effects of Scaling")
            FeatureScaling(data).normalization(column1, column2)
    

    elif engineering_choice == "Handling Missing Values":

        choice = st.multiselect("Select Your Prefered Choice: ", options=["Numeric Value", "Calegorical Value"])

        if "Numeric Value" in choice:
            column = st.selectbox("Select Your Column: ", options=num_columns(data))
            HandlingMissingValues(data).handle_numeric_value(column)
        
        if "Calegorical Value" in choice:
            column = st.selectbox("Select Your Main Categorical Column: ", options=cat_columns(data))
            column2 = st.selectbox("Select Your Optional Numeric Column: ", options=num_columns(data))
            HandlingMissingValues(data).handle_categorical_value(column, column2)
        
    
    elif engineering_choice == "Mathamatical Transformation":
        
        column = st.selectbox("Select Your Column: ", options=num_columns(data))
        MathamaticalTRansformation(data).handle_mathamatical_transformation(input_column=column)