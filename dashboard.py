import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px


st.title("💳 Credit Card Fraud Detection Dashboard")

model = joblib.load("LogReg.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv('creditcard.csv')
df_train_ros = pd.read_csv("train_ros.csv")

st.subheader("Dataset Preview")
st.write(df.head())

X = scaler.transform(df[df.columns[:-1]])

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:,1]

counts = df['Class'].value_counts()
counts_ros = df_train_ros['Class'].value_counts()

total = counts.sum()
total_ros = counts_ros.sum()
percentages = counts / total * 100
percentages_ros = counts_ros / total_ros * 100


fig_orig = px.bar(
    x=['legit', 'fraud'],
    y=percentages,
    text=percentages.round(2),
    title="Original Data Class Distribution (%)"
)

fig_orig.update_traces(textposition='outside')

fig_ros = px.bar(
    x=['legit', 'fraud'],
    y=percentages_ros,
    text=percentages_ros.round(2),
    title="Training Data Class Distribution (%)"
)

fig_ros.update_traces(textposition='outside')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Data")
    st.plotly_chart(fig_orig)

with col2:
    st.subheader("After Oversampling")
    st.plotly_chart(fig_ros)


counts = df_train_ros['Class'].value_counts()

df['Class_label'] = df['Class'].map({0: 'Legit', 1: 'Fraud'})

selected_feature = st.selectbox("Select feature", df.columns[:-1])

fig_featImpor = px.histogram(
    df,
    x=selected_feature,
    color='Class_label',
    nbins=50,
    histnorm='probability density',
    barmode='overlay',
    opacity=0.6,
    title=f"Distribution of {selected_feature}"
)
st.subheader("Feature Importance")
st.plotly_chart(fig_featImpor)

df["Fraud Prediction"] = predictions
df["Fraud Probability"] = probabilities

fraud_count = df["Fraud Prediction"].sum()

st.subheader("Predictions")
st.write(df)
st.metric("Detected Fraud Transactions", fraud_count)