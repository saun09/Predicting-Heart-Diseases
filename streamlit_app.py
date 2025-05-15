import streamlit as st
import joblib
import numpy as np

# Load models
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'KNN': joblib.load('knn_model.pkl'),
    'SVM': joblib.load('svm_model.pkl')
}

# Descriptions for inputs
descriptions = {
    'age': 'Age in years',
    'sex': 'Gender',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (in mm Hg)',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)',
    'restecg': 'Resting electrocardiographic results (0,1,2)',
    'thalch': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0,1,2)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

st.title('Heart Disease Prediction')

# User input collection
age = st.slider('Age', 1, 120, 50)

# Gender dropdown with mapping
gender = st.selectbox('Gender', ['Male', 'Female'])
sex = 1 if gender == 'Male' else 0

# For other features
cp = st.selectbox(f"Chest pain type (cp): {descriptions['cp']}", [0,1,2,3])
trestbps = st.number_input(f"Resting blood pressure (trestbps): {descriptions['trestbps']}", min_value=80, max_value=200, value=120)
chol = st.number_input(f"Serum cholesterol (chol): {descriptions['chol']}", min_value=100, max_value=600, value=200)
fbs = st.selectbox(f"Fasting blood sugar > 120 mg/dl (fbs): {descriptions['fbs']}", [0,1])
restecg = st.selectbox(f"Resting ECG results (restecg): {descriptions['restecg']}", [0,1,2])
thalch = st.number_input(f"Max heart rate achieved (thalch): {descriptions['thalch']}", min_value=60, max_value=220, value=150)
exang = st.selectbox(f"Exercise induced angina (exang): {descriptions['exang']}", [0,1])
oldpeak = st.number_input(f"ST depression (oldpeak): {descriptions['oldpeak']}", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox(f"Slope of ST segment (slope): {descriptions['slope']}", [0,1,2])
ca = st.selectbox(f"Number of major vessels (ca): {descriptions['ca']}", [0,1,2,3])
thal = st.selectbox(f"Thalassemia (thal): {descriptions['thal']}", [1,2,3])

# Collect inputs in array for model
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalch,
                            exang, oldpeak, slope, ca, thal]])

# Model choice dropdown
model_choice = st.selectbox('Select Model', list(models.keys()))

if st.button('Predict'):
    model = models[model_choice]
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1] if hasattr(model, 'predict_proba') else None

    st.write(f"### Prediction: {'Heart Disease Detected' if prediction else 'No Heart Disease Detected'}")
    if proba is not None:
        st.write(f"### Prediction Probability: {proba:.2f}")

