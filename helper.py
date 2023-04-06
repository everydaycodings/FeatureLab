import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import tempfile
import seaborn as sns


def read_data(raw_data):

    data = pd.read_csv(raw_data)

    return data

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


        fig1 = px.scatter(data, x=column1, y=column2, title="Before Sclaing")
        fig2 = px.scatter(data_scaled, x=column1, y=column2, title="After Scaling")

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
        
