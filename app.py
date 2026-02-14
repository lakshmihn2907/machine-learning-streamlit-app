"""

BITS PILANI - ML ASSIGNMENT 2

STREAMLIT APP - BANK MARKETING PREDICTION

"""

import zipfile

import streamlit as st

import pandas as pd

import numpy as np

import joblib

import plotly.express as px

from sklearn.metrics import (

    accuracy_score, roc_auc_score, precision_score,

    recall_score, f1_score, matthews_corrcoef,

    confusion_matrix, classification_report

)

import os


# ============ EXTRACT MODEL ZIP IF NEEDED ============

if not os.path.exists('model/Logistic_Regression.pkl'):

    with zipfile.ZipFile('model.zip', 'r') as zip_ref:

        zip_ref.extractall('.')


# ============ PAGE CONFIG ============

st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

# ============ HEADER ============

st.title("BITS Pilani - Machine Learning Assignment 2")

st.subheader("Bank Marketing Campaign - Term Deposit Prediction")

st.markdown("---")

# ============ SIDEBAR ============

with st.sidebar:

    st.header("Assignment 2")

    st.markdown("**Name:** Lakshmi H N")

    st.markdown("**BITSID:** 2025AB05015")

    st.markdown("**Course:** Machine Learning")

    st.markdown("**Dataset:** UCI Bank Marketing")

    st.markdown("**Models:** 6 Classification Models")

    st.markdown("**Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC")

    st.markdown("---")

    uploaded_file = st.file_uploader("Choose test data CSV", type="csv")

    st.subheader("Select Model")

    model_options = [

        'Logistic Regression',

        'Decision Tree',

        'K-Nearest Neighbors',

        'Naive Bayes',

        'Random Forest',

        'XGBoost'

    ]

    selected_model = st.selectbox("Choose model for prediction", model_options)


# ============ LOAD MODELS ============

@st.cache_resource

def load_model(model_name):

    model_files = {

        'Logistic Regression': 'model/Logistic_Regression.pkl',

        'Decision Tree': 'model/Decision_Tree.pkl',

        'K-Nearest Neighbors': 'model/kNN.pkl',

        'Naive Bayes': 'model/Naive_Bayes.pkl',

        'Random Forest': 'model/Random_Forest.pkl',

        'XGBoost': 'model/XGBoost.pkl'

    }

    model = joblib.load(model_files[model_name])

    scaler = joblib.load('model/scaler.pkl')

    return model, scaler


# ============ MAIN APP ============

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Samples", test_df.shape[0])

    col2.metric("Features", test_df.shape[1])

    col3.metric("Selected Model", selected_model)

    with st.expander("View Uploaded Test Data", expanded=True):

        st.dataframe(test_df.head(), use_container_width=True)

    has_target = 'y' in test_df.columns

    if has_target:

        X_test = test_df.drop('y', axis=1)

        y_true = test_df['y']

    else:

        X_test = test_df.copy()

        y_true = None

    X_test = pd.get_dummies(X_test)

    # ============ PREDICTIONS ============

    if st.button("Make Predictions", use_container_width=True):

        model, scaler = load_model(selected_model)

        # -------- FEATURE ALIGNMENT --------

        if selected_model in ['Logistic Regression', 'K-Nearest Neighbors']:

            if hasattr(scaler, 'feature_names_in_'):

                expected_features = scaler.feature_names_in_

            else:

                expected_features = model.feature_names_in_

        else:

            expected_features = model.feature_names_in_

        # Add missing columns

        for col in expected_features:

            if col not in X_test.columns:

                X_test[col] = 0

        # Remove extra columns & fix order

        X_test = X_test.reindex(columns=expected_features, fill_value=0)

        # -------- PREDICTION --------

        if selected_model in ['Logistic Regression', 'K-Nearest Neighbors']:

            X_scaled = scaler.transform(X_test)

            predictions = model.predict(X_scaled)

            probabilities = model.predict_proba(X_scaled)

        else:

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)

        # -------- RESULTS DISPLAY --------

        st.markdown("---")

        st.subheader("Prediction Results")

        results_df = X_test.copy()

        results_df['Prediction'] = predictions

        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No', 1: 'Yes'})

        results_df['Probability_No'] = probabilities[:, 0].round(4)

        results_df['Probability_Yes'] = probabilities[:, 1].round(4)

        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False)

        st.download_button(

            label="Download Predictions CSV",

            data=csv,

            file_name=f"{selected_model}_predictions.csv",

            mime="text/csv"

        )

        # -------- METRICS --------

        if y_true is not None:

            st.markdown("---")

            st.subheader("Evaluation Metrics")

            acc = accuracy_score(y_true, predictions)

            auc = roc_auc_score(y_true, probabilities[:, 1])

            precision = precision_score(y_true, predictions, zero_division=0)

            recall = recall_score(y_true, predictions, zero_division=0)

            f1 = f1_score(y_true, predictions, zero_division=0)

            mcc = matthews_corrcoef(y_true, predictions)

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{acc:.4f}")

            col1.metric("AUC", f"{auc:.4f}")

            col2.metric("Precision", f"{precision:.4f}")

            col2.metric("Recall", f"{recall:.4f}")

            col3.metric("F1 Score", f"{f1:.4f}")

            col3.metric("MCC Score", f"{mcc:.4f}")

            # Confusion Matrix

            st.markdown("---")

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, predictions)

            fig = px.imshow(

                cm,

                text_auto=True,

                x=['Predicted No', 'Predicted Yes'],

                y=['Actual No', 'Actual Yes'],

                color_continuous_scale='Blues',

                title=f"Confusion Matrix - {selected_model}"

            )

            st.plotly_chart(fig, use_container_width=True)

            # Classification Report

            st.subheader("Classification Report")

            report = classification_report(

                y_true,

                predictions,

                target_names=['No Subscription', 'Subscription'],

                output_dict=True

            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df, use_container_width=True)

else:

    st.info("Please upload test CSV file to begin.")
 
