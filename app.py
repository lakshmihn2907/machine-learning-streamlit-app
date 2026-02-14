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

import plotly.graph_objects as go

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,

                           recall_score, f1_score, matthews_corrcoef,

                           confusion_matrix, classification_report)

import os

# Extract model files if not already extracted
if not os.path.exists('model/Logistic_Regression.pkl'):
   with zipfile.ZipFile('model.zip', 'r') as zip_ref:  
       zip_ref.extractall('.')


# ============ PAGE CONFIG ============

st.set_page_config(

    page_title="BITS ML Assignment 2",

    layout="wide"

)

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

    st.markdown("**Dataset:** UCI Bank Marketing (bank-full.csv)")

    st.markdown("**Models:** 6 Classification Models")

    st.markdown("**Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC")

    st.markdown("---")

    # ============ FEATURE 1: UPLOAD TEST DATA ============

    uploaded_file = st.file_uploader(

        "Choose test data CSV",

        type="csv",

    )

    st.markdown("---")

    # ============ FEATURE 2: MODEL SELECTION ============

    st.subheader("Step 2: Select Model")

    model_options = [

        'Logistic Regression',

        'Decision Tree',

        'K-Nearest Neighbors',

        'Naive Bayes',

        'Random Forest',

        'XGBoost'

    ]

    selected_model = st.selectbox(

        "Choose model for prediction",

        model_options

    )

    st.markdown("---")

    st.markdown("**6 models pre-trained on 45,211 samples**")

    st.markdown("**SMOTE applied for class imbalance**")

# ============ LOAD MODELS ============

@st.cache_resource

def load_model(model_name):

    """Load pre-trained model and scaler"""

    model_files = {

        'Logistic Regression': 'model/Logistic_Regression.pkl',

        'Decision Tree': 'model/Decision_Tree.pkl',

        'K-Nearest Neighbors': 'model/kNN.pkl',

        'Naive Bayes': 'model/Naive_Bayes.pkl',

        'Random Forest': 'model/Random_Forest.pkl',

        'XGBoost': 'model/XGBoost.pkl'

    }

    try:

        model = joblib.load(model_files[model_name])

        scaler = joblib.load('model/scaler.pkl')

        return model, scaler

    except Exception as e:

        st.error(f"Error loading model: {str(e)}")

        return None, None

# ============ MAIN APP ============

if uploaded_file is not None:

    # Load test data

    test_df = pd.read_csv(uploaded_file)

    # Display test data info

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("Test Samples", test_df.shape[0])

    with col2:

        st.metric("Features", test_df.shape[1])

    with col3:

        st.metric("Selected Model", selected_model)

    # Show test data preview

    with st.expander("View Uploaded Test Data", expanded=True):

        st.dataframe(test_df.head(10), use_container_width=True)

    # Check if target column exists

    has_target = 'y' in test_df.columns

    if has_target:

        X_test = test_df.drop('y', axis=1)

        y_true = test_df['y']

    else:

        X_test = test_df.copy()

        y_true = None
      
    X_test = pd.get_dummies(X_test)

    # ============ MAKE PREDICTIONS ============

    if st.button("Make Predictions", use_container_width=True):

        with st.spinner(f"Making predictions using {selected_model}..."):

            model, scaler = load_model(selected_model)

            if model is not None:

                # Preprocess based on model type# ================= FEATURE ALIGNMENT =================

                # Decide expected features

                if selected_model in ['Logistic Regression', 'K-Nearest Neighbors'] and scaler is not None:
                  
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

                 # Remove extra columns and fix order

                 X_test = X_test.reindex(columns=expected_features, fill_value=0)

# ================= PREDICTION =================

            if selected_model in ['Logistic Regression', 'K-Nearest Neighbors'] and scaler is not None:

                  X_test_scaled = scaler.transform(X_test)

                  predictions = model.predict(X_test_scaled)

                  probabilities = model.predict_proba(X_test_scaled)

            else:

                  predictions = model.predict(X_test)

                  probabilities = model.predict_proba(X_test)
 

                              
              
      else:
                  
                   # For XGBoost, Random Forest, Decision Tree, Naive Bayes
                  
                   # Get training feature names from model
          
                   if hasattr(model, 'feature_names_in_'):
                  
                      expected_features = model.feature_names_in_
          
                      # Add missing columns with 0
          
                      for col in expected_features:
                  
                        if col not in X_test.columns:
                  
                           X_test[col] = 0
          
                           # Keep only expected columns in same order
          
                           X_test = X_test[expected_features]
          
                           predictions = model.predict(X_test)
          
                           probabilities = model.predict_proba(X_test)
          
                # ============ DISPLAY PREDICTIONS ============

                st.markdown("---")

                st.subheader("Prediction Results")

                results_df = X_test.copy()

                results_df['Prediction'] = predictions

                results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No', 1: 'Yes'})

                results_df['Probability_No'] = probabilities[:, 0].round(4)

                results_df['Probability_Yes'] = probabilities[:, 1].round(4)

                st.dataframe(results_df, use_container_width=True)

                # Download button

                csv = results_df.to_csv(index=False)

                st.download_button(

                    label="Download Predictions CSV",

                    data=csv,

                    file_name=f"{selected_model}_predictions.csv",

                    mime="text/csv"

                )

                # ============ FEATURE 3: EVALUATION METRICS ============

                if y_true is not None:

                    st.markdown("---")

                    st.subheader("Evaluation Metrics")

                    # Calculate ALL 6 metrics

                    acc = accuracy_score(y_true, predictions)

                    auc = roc_auc_score(y_true, probabilities[:, 1])

                    precision = precision_score(y_true, predictions, zero_division=0)

                    recall = recall_score(y_true, predictions, zero_division=0)

                    f1 = f1_score(y_true, predictions, zero_division=0)

                    mcc = matthews_corrcoef(y_true, predictions)

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric("Accuracy", f"{acc:.4f}")

                        st.metric("AUC", f"{auc:.4f}")

                    with col2:

                        st.metric("Precision", f"{precision:.4f}")

                        st.metric("Recall", f"{recall:.4f}")

                    with col3:

                        st.metric("F1 Score", f"{f1:.4f}")

                        st.metric("MCC Score", f"{mcc:.4f}")

                    # ============ FEATURE 4: CONFUSION MATRIX ============

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

                    fig.update_layout(height=400)

                    st.plotly_chart(fig, use_container_width=True)

                    # Classification Report

                    st.subheader("Classification Report")

                    report = classification_report(

                        y_true, predictions,

                        target_names=['No Subscription', 'Subscription'],

                        output_dict=True

                    )

                    report_df = pd.DataFrame(report).transpose()

                    st.dataframe(report_df.style.format({

                        'precision': '{:.3f}',

                        'recall': '{:.3f}',

                        'f1-score': '{:.3f}',

                        'support': '{:.0f}'

                    }), use_container_width=True)

else:

    # Show instructions

    st.info("**Please upload test data CSV file to begin**")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""

        ###  Instructions

        1. **Upload test data CSV** (bank.csv)

        2. **Select ML model** from dropdown

        3. **Click 'Make Predictions'**

        4. **View results** and download

        ###  Test Data Format

        - Same features as bank-full.csv

        - Include 'y' column for evaluation

        - 100-1000 rows recommended

        """)

    with col2:

        st.markdown("""

        ###  Available Models

         Logistic Regression  

         Decision Tree  

         K-Nearest Neighbors  

         Naive Bayes  

         Random Forest  

         XGBoost  

        ###  Evaluation Metrics

         Accuracy  

         AUC  

         Precision  

         Recall  

         F1 Score  

         MCC Score

        """)

    # Sample test data download

    st.markdown("---")

    st.subheader("📎 Download Sample Test Data")

    sample_data = pd.DataFrame({

        'age': [35, 42, 28, 51, 39],

        'job': ['admin.', 'technician', 'blue-collar', 'management', 'services'],

        'marital': ['married', 'single', 'married', 'divorced', 'married'],

        'education': ['university.degree', 'high.school', 'basic.9y', 'university.degree', 'high.school'],

        'default': ['no', 'no', 'no', 'no', 'no'],

        'balance': [1200, 4500, 800, 3200, 2100],

        'housing': ['yes', 'no', 'yes', 'yes', 'no'],

        'loan': ['no', 'no', 'yes', 'no', 'no'],

        'contact': ['cellular', 'cellular', 'telephone', 'cellular', 'cellular'],

        'day': [15, 22, 8, 12, 19],

        'month': ['may', 'jun', 'jul', 'may', 'jun'],

        'duration': [180, 95, 320, 210, 145],

        'campaign': [1, 2, 1, 3, 1],

        'pdays': [-1, -1, 6, -1, 3],

        'previous': [0, 0, 1, 0, 2],

        'poutcome': ['unknown', 'unknown', 'success', 'unknown', 'failure'],

        'y': [0, 0, 1, 0, 1]

    })

    csv = sample_data.to_csv(index=False)

    st.download_button(

        label="Download sample_test_data.csv",

        data=csv,

        file_name="sample_test_data.csv",

        mime="text/csv"

    )

# ============ FOOTER ============

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
<p> BITS Pilani - Machine Learning Assignment 2 | 15 February 2026</p>
<p> 6 Models |  6 Metrics |  Confusion Matrix |  Streamlit Deployment</p>
</div>

""", unsafe_allow_html=True) 
