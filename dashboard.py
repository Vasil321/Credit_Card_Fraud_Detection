import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import average_precision_score
import xgboost as xgb
import shap


st.title("💳 Credit Card Fraud Detection Dashboard")

model = joblib.load("xgb_ros.pkl")
scaler = joblib.load("scaler.pkl")
X_test, y_test = joblib.load("test_data.pkl")
df_train_ros = pd.read_csv("train_ros.csv")



@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

@st.cache_data
def preprocess_data(df):
    features = df.drop(columns=["Class"], errors="ignore")
    return scaler.transform(features)

df = load_data()
X = preprocess_data(df)

tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🔍 Predictions",
    "📈 Model Performance"
])
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5)
st.sidebar.markdown(f"""
Adjust Threshold
- More strict (higher threshold = fewer fraud alerts)
- More sensitive (lower threshold = more fraud detection)
""")

with tab1:
    st.subheader("Dataset Overview")
    st.info("Columns in tables can be filtered by using the controls provided when hovered over.")
    st.write(df.head())



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
    
    with st.expander("Class Distribution Explanation"):
        st.markdown("""
        **What this shows:**  
        This chart displays the distribution of fraudulent and non-fraudulent transactions in the dataset.

        **How to interpret it:**  
        - Original data (left) has a big class imbalance (0.17'%' fradulent cases).
        - Resampled data (rigt) balances out the class to a 50/50 ratio.
    """)

    counts = df_train_ros['Class'].value_counts()

    df['Class_label'] = df['Class'].map({0: 'Legit', 1: 'Fraud'})

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox("Select feature", df.columns[:-1])

    fig_featDis = px.histogram(
        df,
        x=selected_feature,
        color='Class_label',
        nbins=50,
        histnorm='probability density',
        barmode='overlay',
        opacity=0.6,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig_featDis)
        
    with st.expander("Feature Distribution Explanation"):
        st.markdown("""
        **What this shows:**  
        This plot illustrates the distribution of a selected feature across all transactions.

        **How to interpret it:**  
        Represent the probability of classes (blue = Fraud) (lightblue = begit) depending on the feature
        - Wide spread + high peaks = Good feature distribution
        - Close spread + low peaks = Bad distribution
        """)

    importance = model.get_booster().get_score(importance_type="gain")

    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Importance": list(importance.values())
    })

    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)

    imp_df["Importance"] = imp_df["Importance"].round(2)

    st.subheader("Feature Importance")

    fig = px.bar( 
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=imp_df["Importance"],
        title="Top Feature Importance"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig)
    with st.expander("Feature Importance Explanation"):
        st.markdown("""
        **What this shows:**  
        This chart ranks features based on their importance in the model’s predictions.

        **How to interpret it:**  
        - Higher values indicate features that have a stronger influence on the model  
        - Lower values indicate less impactful features  
        """)



with tab2:
    df["Fraud Probability"] = probabilities
    df["Fraud Prediction"] = (probabilities > threshold).astype(int)


    st.subheader("Predictions")
    st.write(df)

    y_proba = model.predict_proba(X_test)[:,1]

    y_pred = (y_proba > threshold).astype(int)
    pr_auc = average_precision_score(y_test, y_proba)
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Predicted Frauds", int(df["Fraud Prediction"].sum()))
    col3.metric("Max Fraud Risk", f"{df['Fraud Probability'].max():.2%}")

    high_risk = df[df["Fraud Probability"] > 0.8]

    st.subheader("High Risk Transactions")
    st.write(high_risk.sort_values(by="Fraud Probability", ascending=False))


    st.subheader("🔍 SHAP Explanation")

    features = df.drop(
        columns=["Class", "Class_label", "Fraud Prediction", "Fraud Probability"],
        errors="ignore"
    )

    high_risk = df[df["Fraud Probability"] > 0.8]

    selected_index = st.selectbox(
        "Select high-risk transaction",
        high_risk.index
    )

    row = features.loc[[selected_index]]

    explainer = shap.Explainer(model)
    shap_values = explainer(row)


    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    with st.expander("What this shows"):
        st.markdown("""
        SHAP values explain how each feature contributes to a prediction.

        - Positive values = increase fraud likelihood  
        - Negative values = decrease fraud likelihood  
        """)


with tab3:
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['Legit', 'Fraud'],
        y=['Legit', 'Fraud'],
    )

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    fig_PR = go.Figure()

    fig_PR.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='PR Curve'
    ))

    fig_PR.add_annotation(
        x=0.1,
        y=0.0,
        xref="paper",
        yref="paper",
        text=f"PR-AUC = {pr_auc:.2f}",
        showarrow=False
    )

    fig_PR.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    col1_CM, col2_PR = st.columns(2)

    with col1_CM:
        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        with st.expander("Confusion Matrix Explanation"):
            st.markdown("""
            **What this shows:**  
            This confusion matrix illustrates the model's performance in distinguishing between fraudulent and legitimate transactions.

            **How to interpret it:**  
            - Top-left: correctly predicted non-fraud  
            - Bottom-right: correctly predicted fraud  
            - Off-diagonal: errors (false positives/negatives)
            """)

    with col2_PR:
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_PR)
        with st.expander("PR-AUC Explanation"):
            st.write("""
            This precision-recall curve shows the trade-off between precision and recall.
            
            A higher area under the curve indicates better model performance,
            especially important for imbalanced datasets like fraud detection.
            """)

