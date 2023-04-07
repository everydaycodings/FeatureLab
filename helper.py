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
        



class HandlingMissingValues:

    def __init__(self, data):
        self.data = data

    def cca_desc(self):

        st.title(" ")
        st.title("Handling Missing Numeric Values")
        cca_markdown_text = """

            ###### Complete case analysis is a statistical method that only includes cases with complete data in the analysis, and excludes any cases with missing data. This method is simple but can lead to a loss of statistical power and potential bias. Other methods, such as multiple imputation, may be better suited for handling missing data in some situations.

            #### Advantage
            1) Easy to implement as no data manipulation required.
            2) Preserves variable distribution.

            #### Disadvantage
            1) It can exclude a large fraction of the original dataset.
            2) Excluded observations(data) could be infomative for the analysis.
            3) When using your model in production, the model will not know how to handle missing data.

            #### Handling Numeric Value
            1) Distribution of Original Data and Transformed data should be same as much as possible, if distribution doesnot match with each other then you should not apply CCA to that perticular Column.
            2) Check the variance, co-variance of the data after applying Mean/Median Imputation, there should not be drastic change in the value between original, mean and median data.
            3) In case of outliers if there is incrementation of outliers after using Mean/Median Imputaion, then that's a red flag.
            
            #### Handling Category Value
            1) Ratio between Original data and Transformed data should be same as much as possible, if distribution doesnot match with each other then you should not apply CCA to that perticular Column.

            #### When to replace missing values with Mean?
            1) When the distribution of your data is normal

            #### When to replace missing values with Median?
            1) When the distribution is Skewed even a littlebit then use median.
        
        """
        with st.expander("What is Handling Missing Numeric Values"):
            st.markdown(cca_markdown_text)
    
    def handle_numeric_value(self, input_column):

        data = self.data

        self.cca_desc()
        st.subheader("Complete Case Analysis (CCA)")
        st.text("Comparision of Original data with CCA applied data")
        cca_new_data = data[num_columns(data)].dropna()

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        data[input_column].hist(bins=50, ax=ax, density=True, color='red')
        cca_new_data[input_column].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)
        ax.legend(['Original Data', 'Removed Missing Value'])
        ax.set_title('Distribution of {}'.format(input_column))
        ax.set_xlabel('{}'.format(input_column))

        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)
        

        st.text("Comparision of Desity PLot between Original data with CCA applied data")
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        data[input_column].plot.density(color='red')
        cca_new_data[input_column].plot.density(color='green')
        ax.legend(['Original Data', 'Removed Missing Value'])
        ax.set_title('Density Plot of {}'.format(input_column))
        ax.set_xlabel('{}'.format(input_column))
        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)



        st.subheader("Handling Missing Values Using Mean/Median")

        mean_data = data[input_column].mean()
        median_data = data[input_column].median()
        data['median_{}'.format(input_column)] = data[input_column].fillna(median_data)
        data['mean_{}'.format(input_column)] = data[input_column].fillna(mean_data)

        st.text("Comparision of Desity PLot between Original data with Mean/Median data")
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        data[input_column].plot(kind='kde', ax=ax)
        data['median_{}'.format(input_column)].plot(kind='kde', ax=ax, color='red')
        data['mean_{}'.format(input_column)].plot(kind='kde', ax=ax, color='green')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        ax.set_xlabel('{}'.format(input_column))
        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)


        st.text("Change in Outliers after applying Mean/Median Imputation")
        fig = plt.figure(figsize=(12, 5))
        data[['{}'.format(input_column), 'median_{}'.format(input_column), 'mean_{}'.format(input_column)]].boxplot()
        with tempfile.TemporaryDirectory() as path:
            img_path = "{}/save.png".format(path)
            plt.savefig(img_path)
            st.image(img_path)


        st.text("Change In variance after applying Mean/Median Imputation")
        st.markdown("**Original {} Variance: {}**".format(input_column, np.round(data[input_column].var(), 3)))
        st.markdown("**{} Variance after Mean Imputation: {}**".format(input_column, np.round(data["mean_{}".format(input_column)].var(), 3)))
        st.markdown("**{} Variance after Median Imputation: {}**".format(input_column, np.round(data["median_{}".format(input_column)].var(), 3)))
        
        st.text("Change In co-variance after applying Mean/Median Imputation")
        st.dataframe(data.cov())
        

    def handle_categorical_value(self, input_column):

        self.cca_desc()
        st.subheader("Complete Case Analysis (CCA)")

        data = self.data
        cca_new_data = data[cat_columns(data)].dropna()
        
        st.text("Ratio Comparision between Original Data and CCA applied Data(ratio should be same)")
        temp = pd.concat([
            # percentage of observations per category, original data
            data[input_column].value_counts() / len(data),

            # percentage of observations per category, cca data
            cca_new_data[input_column].value_counts() / len(cca_new_data)
        ],
        axis=1)

        # add column names
        temp.columns = ['Original Data', 'CCA Data']
            
        st.dataframe(temp)
