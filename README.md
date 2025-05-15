# Predicting-Heart-Diseases

An AI-driven project for predicting heart disease risk using multiple machine learning models including Logistic Regression, Support Vector Classifier (SVC), and Decision Tree.

---

## 🔍 Model Performance Evaluation

This section provides a detailed comparative analysis of the three classifiers based on evaluation metrics and performance curves.

---

## 📊 Metrics Summary

### 1. Logistic Regression

- **Confusion Matrix:**  
  - True Positives (TP): 89  
  - True Negatives (TN): 67  
  - False Positives (FP): 15  
  - False Negatives (FN): 13  
- **Accuracy:** 0.8478  
- **Precision:** 0.8558  
- **Recall:** 0.8725  
- **F1 Score:** 0.8641  
- **AUC (ROC):** 0.90  
- **Conclusion:** Best overall performance, with a balanced precision and recall, indicating strong generalization capability.

#### Confusion Matrix Image  
`![Logistic Regression Confusion Matrix](path/to/logistic_regression_confusion_matrix.png)`

#### ROC Curve Image  
`![Logistic Regression ROC Curve](path/to/logistic_regression_roc_curve.png)`

---

### 2. Support Vector Classifier (SVC)

- **Confusion Matrix:**  
  - TP: 83  
  - TN: 55  
  - FP: 27  
  - FN: 19  
- **Accuracy:** 0.75  
- **Precision:** 0.7545  
- **Recall:** 0.8137  
- **F1 Score:** 0.7830  
- **AUC (ROC):** 0.78  
- **Conclusion:** Strong recall performance but slightly lower precision, making it more sensitive but prone to some false positives.

#### Confusion Matrix Image  
`![SVC Confusion Matrix](path/to/svc_confusion_matrix.png)`

#### ROC Curve Image  
`![SVC ROC Curve](path/to/svc_roc_curve.png)`

---

### 3. Decision Tree

- **Confusion Matrix:**  
  - TP: 78  
  - TN: 58  
  - FP: 24  
  - FN: 24  
- **Accuracy:** 0.7391  
- **Precision:** 0.7647  
- **Recall:** 0.7647  
- **F1 Score:** 0.7647  
- **AUC (ROC):** 0.74  
- **Conclusion:** Easiest to interpret and visualize, but has the least robust performance among the three.

#### Confusion Matrix Image  
`![Decision Tree Confusion Matrix](path/to/decision_tree_confusion_matrix.png)`

#### ROC Curve Image  
`![Decision Tree ROC Curve](path/to/decision_tree_roc_curve.png)`

---

## Final Verdict

| Metric         | Logistic Regression | Support Vector Classifier (SVC) | Decision Tree   |
|----------------|---------------------|---------------------------------|-----------------|
| Accuracy       | 0.85                | 0.75                            | 0.74            |
| Precision      | 0.86                | 0.75                            | 0.76            |
| Recall         | 0.87                | 0.81                            | 0.76            |
| F1 Score       | 0.86                | 0.78                            | 0.76            |
| AUC (ROC)      | 0.90                | 0.78                            | 0.74            |

---

## Project Summary

- **Goal:** To predict the presence of heart disease based on clinical features using machine learning.
- **Data:** Clinical data with features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, and more.
- **Models Used:** Logistic Regression, Support Vector Classifier, Decision Tree.
- **Tools & Libraries:** Python, scikit-learn, Streamlit (for the interactive app), joblib (for saving/loading models).

---

## How to Use the App

1. Clone the repository.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
