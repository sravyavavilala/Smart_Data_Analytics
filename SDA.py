import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import os

def perform_eda(df):
    st.write("## Basic Statistics")
    st.write(df.describe())
    st.write("## Missing Values")
    st.write(df.isnull().sum())
    st.write("## Data Types")
    st.write(df.dtypes)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        st.write("## Correlation Matrix")
        correlation_matrix = numeric_df.corr()
        st.write(correlation_matrix)
        st.write("## Histograms")
        df[numeric_df.columns].hist(bins=30, figsize=(20, len(numeric_df.columns) * 2))
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for EDA.")

def visualize_data(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        st.write("## Pairplot")
        sns.pairplot(numeric_df)
        st.pyplot(plt)
        st.write("## Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for visualization.")

def detect_anomalies(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        contamination = st.slider("Select contamination level", 0.01, 0.2, 0.05)
        iso_forest = IsolationForest(contamination=contamination)
        df["anomaly"] = iso_forest.fit_predict(numeric_df)
        anomalies = df[df["anomaly"] == -1]
        st.write(f"### Detected {len(anomalies)} anomalies")
        st.write(anomalies)
        st.write("### Anomaly Visualization")
        plt.figure(figsize=(10, 5))
        if len(numeric_df.columns) >= 2:
            sns.scatterplot(
                x=numeric_df.columns[0],
                y=numeric_df.columns[1],
                data=df,
                hue=df["anomaly"],
                palette={1: "blue", -1: "red"}
            )
            st.pyplot(plt.gcf())
        else:
            st.write("Not enough numeric columns for anomaly visualization.")
    else:
        st.write("No numeric columns available for anomaly detection.")

def main():
    st.title("Smart Data Analytics with Anomaly Detection")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        st.write(f"### Number of rows: {df.shape[0]}")
        if st.button("Perform EDA"):
            perform_eda(df)
        if st.button("Visualize Data"):
            visualize_data(df)
        if st.button("Detect Anomalies"):
            detect_anomalies(df)

if __name__ == "__main__":
    main()
