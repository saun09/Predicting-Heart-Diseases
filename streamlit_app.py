import streamlit as st
import joblib
import numpy as np
import pandas as pd

models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'KNN': joblib.load('knn_model.pkl'),
    'SVM': joblib.load('svm_model.pkl')
}
scaler = joblib.load('scaler.pkl')

def main():
    st.title("Heart Disease Prediction App")

    st.write("Please enter the following details:")

    age = st.slider('Age (years)', 20, 100, 55)
    sex = st.selectbox('Gender', ['Male', 'Female'])
    dataset = st.text_input('Dataset Source (optional)', '')  # Can be ignored or removed if not used
    cp = st.selectbox('Chest Pain Type (cp)',
                      ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['Yes', 'No'])
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)',
                          ['Normal', 'Having ST-T wave abnormality', 'Left ventricular hypertrophy'])
    thalch = st.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina?', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    model_name = st.selectbox('Choose Model', list(models.keys()))

    if st.button('Predict'):
        # Map categorical inputs to numeric based on your dataset encoding

        sex_num = 1 if sex == 'Male' else 0

        cp_dict = {
            'Typical Angina': 0,
            'Atypical Angina': 1,
            'Non-anginal Pain': 2,
            'Asymptomatic': 3
        }
        cp_num = cp_dict[cp]

        fbs_num = 1 if fbs == 'Yes' else 0

        restecg_dict = {
            'Normal': 0,
            'Having ST-T wave abnormality': 1,
            'Left ventricular hypertrophy': 2
        }
        restecg_num = restecg_dict[restecg]

        exang_num = 1 if exang == 'Yes' else 0

        slope_dict = {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        }
        slope_num = slope_dict[slope]

        thal_dict = {
            'Normal': 1,
            'Fixed Defect': 2,
            'Reversible Defect': 3
        }
        thal_num = thal_dict[thal]

        # Construct feature array in order expected by the model
        features = np.array([[age, sex_num, cp_num, trestbps, chol,
                              fbs_num, restecg_num, thalch, exang_num,
                              oldpeak, slope_num, ca, thal_num]])

        features_scaled = scaler.transform(features)

        model = models[model_name]

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        result = 'Disease' if prediction == 1 else 'No Disease'
        st.write(f"### Prediction: **{result}**")

        prob_df = pd.DataFrame({
            'Class': ['No Disease', 'Disease'],
            'Probability': probabilities
        })

        st.bar_chart(prob_df.set_index('Class'))

        max_prob = max(probabilities)
        st.write(f"The model predicts **{result}** with a confidence of {max_prob:.2f}.")

        if max_prob < 0.6:
            st.warning("The prediction confidence is low, please consider consulting a medical professional for further diagnosis.")
        else:
            st.info("The model is fairly confident in this prediction based on the input values.")

if __name__ == "__main__":
    main()
