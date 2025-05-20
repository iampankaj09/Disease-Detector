import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load datasets
def load_data(dataset):
    if dataset == "diabetes":
        return pd.read_csv(r"C:\\Users\\Pankaj\\OneDrive\\Desktop\\ML_Dataset\\diseasedetection\\diabeties.csv")
    elif dataset == "heart":
        return pd.read_csv(r"C:\\Users\\Pankaj\\OneDrive\\Desktop\\ML_Dataset\\diseasedetection\\heart.csv")

# Train model
def train_model(data, target_column):
    X = data.iloc[:, :-1]  # Features
    y = data[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Predict function
def predict(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit App
st.title("Disease Detection App")
st.write("Detect diseases using patient data")

# Disease selection
disease = st.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease"])

# Load and display dataset
data = load_data("diabetes" if disease == "Diabetes" else "heart")
st.subheader(f"{disease} Dataset Preview")
st.dataframe(data.head())

# Train model
target_column = "Outcome" if disease == "Diabetes" else "Target"
model, accuracy = train_model(data, target_column)
st.write(f"Model trained with an accuracy of {accuracy:.2f}")

# Input features for prediction
st.subheader("Enter Patient Details")
if disease == "Diabetes":
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
else:  # Heart Disease
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)", [0, 1])
    restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1: Yes, 0: No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3, value=2)
    input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Predict button   
if st.button("Predict"):
    result = predict(model, input_features)
    if disease == "Diabetes":
        st.subheader(f"Prediction: {'Diabetic' if result == 1 else 'Non-Diabetic'}")
    else:
        st.subheader(f"Prediction: {'Heart Disease Detected' if result == 1 else 'No Heart Disease'}")
