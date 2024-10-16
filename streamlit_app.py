import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# import pillow as pillow 
import seaborn as sns
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt

from PIL import Image


st.title("Midterm Project")

app_page = st.sidebar.selectbox('Select Page', ['Landing page/describing data', 'Visualization', 'Prediction'])

# df = pd.read_csv("whr2023.csv")
df = pd.read_csv("whr2023.csv")
if app_page == 'Landing page/describing data':
    
    
    st.write("Welcome to world happiness dashboard!")
    st.subheader("Overview")
    st.write("This dataset is entitled 'World Happiness Report'.")

    st.subheader("Dataset: ")
    st.dataframe(df.head())
    st.write("The World Happiness Report reports data from 2023. A preview of the dataset is shown below: ")
    
    #df2 = df.drop(["Explained by: Log GDP per capita", "Explained by: Social support", "Explained by: Healthy life expectancy", "Explained by: Freedom to make life choices", "Explained by: Generosity", "Explained by: Perception of corruption", "Dystopia + residual"])
    #st.dataframe(df2.head())

    st.write("Source: https://www.kaggle.com/datasets/atom1991/world-happiness-report-2023 ")

    st.subheader("Our Goals")
    st.write("The goal of our project is to analyze the factors which effect a person's life expectancy. With this set of data, we will try to predict people's life expectancy based on numerous factors, such as their happiness score, their country, the gdp associated with their country, ....")

if app_page == 'Visualization':
    st.subheader("03. Data Visualization")

    list_columns = df.columns

    values = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Healthy life expectancy"])

    #Creation of the line chart
    st.line_chart(df, x=values[0], y=values[1])


    string_columns = list(df.select_dtypes(include=['object']).columns)
    data1 = df.drop(columns=string_columns)
    data = data1.drop(columns = 'Ladder score in Dystopia')

    # Create heatmap data (assuming 'value_column' is the column you want to visualize)
    heatmap_data = data.corr()  # Calculate correlation matrix

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", ax=ax, cmap='RdBu_r', vmin=-1, vmax=1) 
    plt.xlabel("X-axis Label", fontsize=12)
    plt.ylabel("Y-axis Label", fontsize=12)
    plt.title("Correlation Matrix", fontsize=14)
    st.pyplot(fig)
# Pairplot
    values_pairplot = st.multiselect("Select 4 variables: ", list_columns, ["Healthy life expectancy", "Happiness score", "Logged GDP per capita", "Freedom to make life choices"])
    df2 = df[[values_pairplot[0], values_pairplot[1], values_pairplot[2], values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)


list_columns = df.columns


if app_page == 'Prediction':
    st.title("03. Prediction")
        
    list_columns = df.columns

    input_lr = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Logged GDP per capita"])

    df_new = df.dropna() 
    df2 = df_new[input_lr]

    X = df2
    y = df_new["Healthy life expectancy"]

    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25 ")
    col1.write(X.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    explained_variance = np.round(mt.explained_variance_score(y_test, pred) * 100, 2)
    mae = np.round(mt.mean_absolute_error(y_test, pred), 2)
    mse = np.round(mt.mean_squared_error(y_test, pred), 2)
    r_square = np.round(mt.r2_score(y_test, pred), 2)

    # Display results
    st.subheader('🎯 Results')
    st.write("1) The model explains,", explained_variance, "% variance of the target feature")
    st.write("2) The Mean Absolute Error of the model is:", mae)
    st.write("3) MSE: ", mse)
    st.write("4) The R-Square score of the model is", r_square)

    # Plotting the Linear Regression line
    st.subheader('📈 Linear Regression Line')

    # Create a scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_new[input_lr[0]], y=y, data=df_new, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    plt.title(f'Linear Regression of Healthy Life Expectancy vs {input_lr[0]}')
    plt.xlabel(input_lr[0])
    plt.ylabel('Healthy Life Expectancy')
    
    st.pyplot(plt)  # Display the plot in Streamlit






