# machine-learning-streamlit-app
BITS Pilani Machine Learning Assignment 2 - Classification Models with Streamlit Deployment
# BITS Pilani - Machine Learning Assignment 2

**Name:** Lakshmi H N  

**Course:** Machine Learning  

**Dataset:** UCI Bank Marketing (bank-full.csv)  

**Submission:** 15 February 2026

---

##  Problem Statement

Predict whether a client will subscribe to a term deposit based on marketing campaign data. Implement and compare 6 classification models using 6 evaluation metrics.

---

##  Dataset Description

| Attribute | Value |

|-----------|-------|

| **Source** | UCI Machine Learning Repository |

| **Instances** | 45,211 |

| **Features** | 20 (after preprocessing) |

| **Target** | Binary (yes/no for term deposit) |

| **Class Imbalance** | 11.7% positive / 88.3% negative |

---

##  Models Implemented

1. Logistic Regression

2. Decision Tree

3. K-Nearest Neighbors (kNN)

4. Naive Bayes (Gaussian)

5. Random Forest (Ensemble)

6. XGBoost (Ensemble)

---

##  Model Performance Comparison

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|----------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression  | 0.9057   | 0.9108 | 0.5822    | 0.5765 | 0.5793   | 0.5262 |
| Decision Tree        | 0.8886   | 0.9033 | 0.5036    | 0.7522 | 0.6033   | 0.5562 |
| K-Nearest Neighbors  | 0.8900   | 0.8172 | 0.5143    | 0.4256 | 0.4658   | 0.4074 |
| Naive Bayes          | 0.8593   | 0.8269 | 0.3979    | 0.4849 | 0.4371   | 0.3599 |
| Random Forest        | 0.9142   | 0.9456 | 0.6172    | 0.6272 | 0.6221   | 0.5737 |
| XGBoost              | 0.9165   | 0.9498 | 0.6279    | 0.6347 | 0.6313   | 0.5842 |

** Best Overall: XGBoost** (Highest Accuracy, AUC, Precision, F1, MCC)

---

##  Observations on Model Performance

| ML Model Name             | Observation about Model Performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | Strong baseline with 90.57% accuracy. Good AUC (0.9108) shows excellent ranking ability. Balanced precision (0.5822) and recall (0.5765). F1 Score (0.5793) and MCC (0.5262) indicate stable overall performance. |
| Decision Tree             | Accuracy of 88.86% with highest recall (0.7522), meaning it captures most actual subscribers. AUC (0.9033) is strong. Precision (0.5036), F1 (0.6033), and MCC (0.5562) show good but slightly less balanced performance. |
| K-Nearest Neighbors (kNN) | Moderate performance with 89.00% accuracy. Lower AUC (0.8172) indicates weaker ranking ability. Precision (0.5143), recall (0.4256), F1 (0.4658), and MCC (0.4074) show room for improvement. |
| Naive Bayes               | Fastest training but lowest accuracy (85.93%). AUC (0.8269) is moderate. Lower precision (0.3979), recall (0.4849), F1 (0.4371), and MCC (0.3599) suggest independence assumption may not fully hold. |
| Random Forest (Ensemble)  | Excellent performance with 91.42% accuracy. High AUC (0.9456) indicates strong ranking capability. Balanced precision (0.6172) and recall (0.6272). Strong F1 (0.6221) and MCC (0.5737) reflect robust overall classification. |
| XGBoost (Ensemble)        | Best performing model across all metrics. Highest accuracy (91.65%), AUC (0.9498), precision (0.6279), recall (0.6347), F1 (0.6313), and MCC (0.5842). Most balanced and powerful model. |

---

##  Deployment

**Live App:** [https://lakshmihn2907-machine-learning-streamlit-app-app-le83o8.streamlit.app/]  

**GitHub:** [https://github.com/lakshmihn2907/bits-ml-assignment-2](https://github.com/lakshmihn2907/bits-ml-assignment-2)

---

##  Features Implemented

- ✓ 6 Classification Models

- ✓ 6 Evaluation Metrics (Accuracy, AUC, Precision, Recall, F1, MCC)

- ✓ Streamlit App with Test Data Upload

- ✓ Model Selection Dropdown

- ✓ Confusion Matrix Visualization

- ✓ BITS Virtual Lab Execution

---

##  Repository Structure

```

bits-ml-assignment-2/

├── app.py

├── requirements.txt

├── README.md

└── model/

├── Logistic_Regression.pkl

├── Decision_Tree.pkl

├── kNN.pkl

├── Naive_Bayes.pkl

├── Random_Forest.pkl

├── XGBoost.pkl

└── scaler.pkl

```
 
 
