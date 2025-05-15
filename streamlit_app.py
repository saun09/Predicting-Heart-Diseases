import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load saved models and scaler
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

        features = np.array([[age, sex_num, cp_num, trestbps, chol,
                              fbs_num, restecg_num, thalch, exang_num,
                              oldpeak, slope_num, ca, thal_num]])

        features_scaled = scaler.transform(features)

        model = models[model_name]

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Show prediction
        result = 'Disease' if prediction == 1 else 'No Disease'
        st.write(f"### Prediction: **{result}**")

        # Show probabilities bar chart
        prob_df = pd.DataFrame({
            'Class': ['No Disease', 'Disease'],
            'Probability': probabilities
        })

        st.bar_chart(prob_df.set_index('Class'))

        max_prob = max(probabilities)
        st.write(f"The model predicts **{result}** with a confidence of {max_prob:.2f}.")

        if max_prob < 0.6:
            st.warning("The prediction confidence is low, please consider consulting a medical professional for further diagnosis.-This is just a project model - it can make mistakes :) ")
        else:
            st.info("The model is fairly confident in this prediction based on the input values.")

        # Additional feature-based risk factor visualization

        st.subheader("Feature Analysis")

        def color_value(val, low, high):
            if val < low:
                return '🔴 Low'
            elif val > high:
                return '🔴 High'
            else:
                return '🟢 Normal'

        feature_data = {
            'Resting BP (mm Hg)': (trestbps, 90, 120),
            'Cholesterol (mg/dl)': (chol, 125, 200),
            'Max Heart Rate Achieved': (thalch, 100, 170),
            'ST Depression (oldpeak)': (oldpeak, 0, 1),
            'Number of Major Vessels': (ca, 0, 1),
        }

        feature_status = {k: color_value(v[0], v[1], v[2]) for k, v in feature_data.items()}

        risk_df = pd.DataFrame({
            'Feature': list(feature_data.keys()),
            'Value': [v[0] for v in feature_data.values()],
            'Status': [feature_status[k] for k in feature_data.keys()]
        })

        bar_colors = []
        for status in risk_df['Status']:
            if 'High' in status or 'Low' in status:
                bar_colors.append('red')
            else:
                bar_colors.append('green')

        st.write("### Risk Factor Visualization")
        st.write("Red bars indicate values outside the normal range (potential risk factors).")

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(risk_df['Feature'], risk_df['Value'], color=bar_colors)
        ax.set_ylim(0, max(risk_df['Value'].max() * 1.2, 10))
        ax.set_ylabel("Value")

        plt.xticks(rotation=45, ha='right', fontsize=10)

        for bar, status in zip(bars, risk_df['Status']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, status, ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
