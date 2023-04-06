import streamlit as st
from helper import FeatureScaling, read_data, update_secondary_col, num_columns


st.header("Welcome to FeatureLab")

upload_file = st.file_uploader("Upload your data: ")
engineering_choice = st.selectbox("Select your Feature Engineering Method: ", options=["Feature Scaling"])

if upload_file is not None:
    data = read_data(upload_file)
    st.subheader(" ")
    if engineering_choice == "Feature Scaling":
        st.subheader("You Selected Feature Scaling")
        choice = st.multiselect("Select Your Prefered Choice: ", options=["Standardization", "Normalization"])
        
        if "Standardization" in choice:
            st.subheader("Effect of Scaling")
            column1 = st.selectbox("Select Your First Column: ", options=num_columns(data))
            column2 = st.selectbox("Select Your Second Column: ", options=update_secondary_col(num_columns(data), column1))
            FeatureScaling(data).steandardization(column1, column2)