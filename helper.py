import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
import tempfile
import seaborn as sns
import numpy as np

def read_data(raw_data):
        
    return pd.read_csv(raw_data)



def cat_columns(data):
    return [feature for feature in data.columns if data[feature].dtypes == "O"]

def num_columns(data):
    return [feature for feature in data.columns if data[feature].dtypes != "O"]


def update_secondary_col(column, primary_value):

    column.remove(primary_value)

    return column

class FeatureScaling:

    def __init__(self, data):

        self.data = data

    def steandardization(self, column1, column2):

        columns = num_columns(self.data)
        data = self.data[columns]

        scaler = StandardScaler()
        scaler.fit(data)

        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        des1, des2 = st.columns(2)
        with des1:
            st.text("Before Scaling")
            st.dataframe(np.round(data.describe(), 1))
        with des2:
            st.text("After Standard Scaling")
            st.dataframe(np.round(data_scaled.describe(), 1))

        fig1 = px.scatter(data, x=column1, y=column2, title="Before Sclaing")
        fig2 = px.scatter(data_scaled, x=column1, y=column2, title="After Standard Scaling")

        with tempfile.TemporaryDirectory() as path:
            img_path1 = "{}/save.png".format(path)
            img_path2 = "{}/save1.png".format(path)
            fig1.write_image(img_path1)
            fig2.write_image(img_path2)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_path1)
            with col2:
                st.image(img_path2)
        

        st.subheader("Graph For Probability Density Function(pdf) ")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # before scaling
        ax1.set_title('Before Scaling')
        sns.kdeplot(data[column1], ax=ax1)
        sns.kdeplot(data[column2], ax=ax1)

        # after scaling
        ax2.set_title('After Standard Scaling')
        sns.kdeplot(data_scaled[column1], ax=ax2)
        sns.kdeplot(data_scaled[column2], ax=ax2)

        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)
        

        st.subheader("Comparison of Distributions")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # before scaling
        ax1.set_title('{} Distribution Before Scaling'.format(column1))
        sns.kdeplot(data[column1], ax=ax1)

        # after scaling
        ax2.set_title('{} Distribution After Standard Scaling'.format(column1))
        sns.kdeplot(data_scaled[column1], ax=ax2)
        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)
        
        markdown_text = """
            1) Standardization is a technique used in feature scaling to transform the features of a dataset to have a mean of 0 and a standard deviation of 1. This process involves subtracting the mean of the feature from each value in the feature and then dividing by the standard deviation.

            2) Standardization is useful in machine learning because it ensures that all features are on the same scale, making it easier to compare them and preventing one feature from dominating others. This can improve the performance of some machine learning algorithms that are sensitive to the scale of the features.

            3) Standardization is different from other feature scaling techniques, such as normalization or min-max scaling, which scale the features to a specific range, such as between 0 and 1.

            4) Standardization, as a technique of feature scaling, has several advantages in machine learning, including preventing features from dominating others, improving convergence, making features easier to interpret, working well with many machine learning algorithms, and handling outliers better than some other scaling techniques.
        """
        with st.expander("What is Standardization?"):
            st.markdown(markdown_text)
    

    def normalization(self, column1, column2):

        columns = num_columns(self.data)
        data = self.data[columns]

        scaler = MinMaxScaler()
        scaler.fit(data)

        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        des1, des2 = st.columns(2)
        with des1:
            st.text("Before Scaling")
            st.dataframe(np.round(data.describe(), 1))
        with des2:
            st.text("After Normalization Scaling")
            st.dataframe(np.round(data_scaled.describe(), 1))

        fig1 = px.scatter(data, x=column1, y=column2, title="Before Sclaing")
        fig2 = px.scatter(data_scaled, x=column1, y=column2, title="After Normalization Scaling")

        with tempfile.TemporaryDirectory() as path:
            img_path1 = "{}/save.png".format(path)
            img_path2 = "{}/save1.png".format(path)
            fig1.write_image(img_path1)
            fig2.write_image(img_path2)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_path1)
            with col2:
                st.image(img_path2)
        

        st.subheader("Graph For Probability Density Function(pdf) ")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # before scaling
        ax1.set_title('Before Scaling')
        sns.kdeplot(data[column1], ax=ax1)
        sns.kdeplot(data[column2], ax=ax1)

        # after scaling
        ax2.set_title('After Normalization Scaling')
        sns.kdeplot(data_scaled[column1], ax=ax2)
        sns.kdeplot(data_scaled[column2], ax=ax2)

        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)
        

        st.subheader("Comparison of Distributions")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # before scaling
        ax1.set_title('{} Distribution Before Scaling'.format(column1))
        sns.kdeplot(data[column1], ax=ax1)

        # after scaling
        ax2.set_title('{} Distribution After Standard Scaling'.format(column1))
        sns.kdeplot(data_scaled[column1], ax=ax2)
        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)
        
        markdown_text = """
                    Normalization is a feature scaling technique that rescales the features to have values between 0 and 1.

                    Advantages:

                        1) Rescaling the features to a common scale, making them easier to compare.
                        2) Preserves the shape of the original distribution.
                        3) Prevents features with larger values from dominating other features.
                        4) Can improve the performance of some machine learning algorithms, such as those that rely on distance measures, by avoiding the issue of some features being measured on a larger scale than others.

                    Disadvantages:

                        1) Sensitive to outliers, which can affect the normalization of the entire feature.
                        2) Assumes a linear relationship between the features and the target variable, and may not be suitable for non-linear relationships.
                        3) May not be appropriate for some applications where the original scale of the features is meaningful, such as image or signal processing.
            """
        
        with st.expander("What is Normalization?"):
            st.markdown(markdown_text)
        
