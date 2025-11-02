import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump, load
import os

# Load or train the model
@st.cache_resource
def load_model():
    try:
        model = load('best_rf_model.joblib')
        scaler = load('scaler.joblib')
        feature_columns = load('feature_columns.joblib')
        label_encoder = load('label_encoder.joblib')
        return model, scaler, feature_columns, label_encoder
    except FileNotFoundError:
        st.error("Model assets not found! Training a new model...")
        return train_model()

def train_model():
    # Load data
    df = pd.read_csv('travel_data.csv')

    # Encode target
    le = LabelEncoder()
    df['Destination_Encoded'] = le.fit_transform(df['Target_Destination'])

    X = df.drop(['Target_Destination', 'Destination_Encoded'], axis=1)
    y = df['Destination_Encoded']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=['Gender', 'Income_Level', 'Travel_Companion', 'Budget'], drop_first=True)

    # Scale numerical features
    numerical_features = ['Age', 'Activity_Level']
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    # Save model assets
    dump(model, 'best_rf_model.joblib')
    dump(scaler, 'scaler.joblib')
    dump(X_train.columns.tolist(), 'feature_columns.joblib')
    dump(le, 'label_encoder.joblib')

    return model, scaler, X_train.columns.tolist(), le

# Load model
model, scaler, feature_columns, label_encoder = load_model()

# Streamlit app
st.title("Personalized Travel Destination Recommender")
st.markdown("Powered by Tuned Random Forest Classifier")
st.markdown("Tell us about your preferences and we'll recommend the perfect destination!")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])

with col2:
    travel_companion = st.selectbox("Travel Companion", ["Solo", "Family", "Group"])
    activity_level = st.slider("Activity Level (1=Relaxing, 5=Intense)", 1, 5, 4)
    budget = st.selectbox("Budget", ["Economical", "Mid-range", "Luxury"])

# Prediction
if st.button("Get Recommendation"):
    # Prepare input
    input_data = pd.DataFrame({
        'Age': [age],
        'Activity_Level': [activity_level],
        'Gender': [gender],
        'Income_Level': [income_level],
        'Travel_Companion': [travel_companion],
        'Budget': [budget]
    })

    # One-hot encode
    input_encoded = pd.get_dummies(input_data, columns=['Gender', 'Income_Level', 'Travel_Companion', 'Budget'], drop_first=True)

    # Scale numerical features
    numerical_features = ['Age', 'Activity_Level']
    input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])

    # Ensure all columns are present
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].iloc[0]

    # Predict
    prediction_encoded = model.predict(final_input)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = model.predict_proba(final_input)[0]
    confidence = probabilities[prediction_encoded] * 100

    # Display result
    emoji_map = {
        "Adventure/Nature": "🏔️ Adventure/Nature",
        "Relaxation/Beach": "🏖️ Relaxation/Beach",
        "Cultural/Historical": "🏛️ Cultural/Historical",
        "Party/Nightlife": "🎉 Party/Nightlife"
    }

    st.success(f"Recommended Destination: {emoji_map.get(prediction_label, '🌍 Unknown')}")
    st.info(f"Model Confidence: {confidence:.2f}%")

    # Additional info
    st.markdown("### Why this destination?")
    if prediction_label == "Adventure/Nature":
        st.write("Based on your high activity level and preference for outdoor experiences!")
    elif prediction_label == "Relaxation/Beach":
        st.write("Perfect for unwinding with your chosen budget and companion!")
    elif prediction_label == "Cultural/Historical":
        st.write("Ideal for exploring history and culture at your pace!")
    elif prediction_label == "Party/Nightlife":
        st.write("Get ready for vibrant nightlife and social adventures!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn")
