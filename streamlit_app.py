import streamlit as st
import joblib
import numpy as np

# Load your saved models and scaler
models = {
    'Logistic Regression': joblib.load('logistic_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'KNN': joblib.load('knn_model.pkl'),
    'SVM': joblib.load('svm_model.pkl'),
}
scaler = joblib.load('scaler.pkl')

# Features list (all numerical)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

st.title("Heart Disease Prediction")

st.write("Enter values for the following features:")

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", format="%.4f")
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

model_name = st.selectbox("Select Model", list(models.keys()))

if st.button("Predict"):
    # Scale input data before prediction
    scaled_input = scaler.transform(input_array)
    
    model = models[model_name]
    prediction = model.predict(scaled_input)[0]
    st.write(f"Prediction: {prediction}")
