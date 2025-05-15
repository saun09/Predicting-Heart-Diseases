# Predicting-Heart-Diseases

A project for predicting heart disease risk using multiple machine learning models including Logistic Regression, Support Vector Classifier (SVC), and Decision Tree.

---

## Metrics Summary

### 1. Logistic Regression
- **Accuracy:** 0.8478  
- **Precision:** 0.8558  
- **Recall:** 0.8725  
- **F1 Score:** 0.8641  
- **AUC (ROC):** 0.90  
Best overall performance, with a balanced precision and recall, indicating strong generalization capability.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f96a974c-796e-49f8-b536-3e0d3629a3e2" width="45%" alt="Logistic Regression Confusion Matrix" />
  <img src="https://github.com/user-attachments/assets/99c9c014-7869-4337-96e2-da399bf0e88a" width="45%" alt="Logistic Regression ROC Curve" />
</div>

<div align="center" style="margin-top:10px;">
  <img src="https://github.com/user-attachments/assets/26575d83-2c73-4dee-a97a-6afdaad98948" width="45%" alt="Logistic Regression Precision Recall Curve" />
</div>

---

### 2. Support Vector Classifier (SVC) 
- **Accuracy:** 0.75  
- **Precision:** 0.7545  
- **Recall:** 0.8137  
- **F1 Score:** 0.7830  
- **AUC (ROC):** 0.78  
Strong recall performance but slightly lower precision, making it more sensitive but prone to some false positives.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e27870b5-7b44-4f8d-9106-a72e97685d43" width="45%" alt="SVC Confusion Matrix" />
  <img src="https://github.com/user-attachments/assets/3020e250-118e-4404-9e51-b4a8fa6c7aa8" width="45%" alt="SVC ROC Curve" />
</div>

<div align="center" style="margin-top:10px;">
  <img src="https://github.com/user-attachments/assets/532cd341-83bc-4f88-8036-0560ddd60fb0" width="45%" alt="SVC Precision Recall Curve" />
</div>

---

### 3. Decision Tree
- **Accuracy:** 0.7391  
- **Precision:** 0.7647  
- **Recall:** 0.7647  
- **F1 Score:** 0.7647  
- **AUC (ROC):** 0.74  
Easiest to interpret and visualize, but has the least robust performance among the three.

<div align="center">
  <img src="https://github.com/user-attachments/assets/6784e14b-2362-41f4-81a1-3964d398c942" width="45%" alt="Decision Tree Confusion Matrix" />
  <img src="https://github.com/user-attachments/assets/1b9daf98-7c5e-4c56-ab7f-9771c8830ad7" width="45%" alt="Decision Tree ROC Curve" />
</div>

<div align="center" style="margin-top:10px;">
  <img src="https://github.com/user-attachments/assets/cf986faf-d871-4e0f-a41b-0a1c846f13e2" width="45%" alt="Decision Tree Precision Recall Curve" />
</div>

---

## Comparison Table

| Metric    | Logistic Regression | Support Vector Classifier (SVC) | Decision Tree |
|-----------|---------------------|---------------------------------|---------------|
| Accuracy  | 0.85                | 0.75                            | 0.74          |
| Precision | 0.86                | 0.75                            | 0.76          |
| Recall    | 0.87                | 0.81                            | 0.76          |
| F1 Score  | 0.86                | 0.78                            | 0.76          |
| AUC (ROC) | 0.90                | 0.78                            | 0.74          |

---

## Project Summary

- **Goal:** To predict the presence of heart disease based on clinical features using machine learning.  
- **Data:** Clinical data with features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, and more.  
- **Models Used:** Logistic Regression, Support Vector Classifier, Decision Tree.  
- **Tools & Libraries:** Python, scikit-learn, Streamlit (for the interactive app), joblib (for saving/loading models).

---

## Try it out on:  
https://predicting-heart-diseases-saundarya.streamlit.app/

<div align="center">
  <img src="https://github.com/user-attachments/assets/00d17fed-ddd7-4303-884a-db5d39395e6b" width="45%" alt="App Screenshot 1" />
  <img src="https://github.com/user-attachments/assets/bc4e7645-19b3-45a3-9ede-5452d3ff84f7" width="45%" alt="App Screenshot 2" />
</div>

<div align="center" style="margin-top:10px;">
  <img src="https://github.com/user-attachments/assets/254289f1-0106-4ece-8ea9-342759a358bf" width="45%" alt="App Screenshot 3" />
  <img src="https://github.com/user-attachments/assets/d4da71da-191e-4604-a6d3-d3cae9c405b7" width="45%" alt="App Screenshot 4" />
</div>
