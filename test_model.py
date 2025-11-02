import pandas as pd
import numpy as np
from joblib import load

# Load model assets
model = load('best_rf_model.joblib')
scaler = load('scaler.joblib')
feature_columns = load('feature_columns.joblib')
label_encoder = load('label_encoder.joblib')

# Test with different inputs
test_inputs = [
    {'Age': 25, 'Gender': 'Male', 'Income_Level': 'Low', 'Travel_Companion': 'Solo', 'Activity_Level': 5, 'Budget': 'Economical'},  # Should be Adventure
    {'Age': 50, 'Gender': 'Female', 'Income_Level': 'High', 'Travel_Companion': 'Family', 'Activity_Level': 1, 'Budget': 'Luxury'},  # Should be Relaxation
    {'Age': 22, 'Gender': 'Other', 'Income_Level': 'Medium', 'Travel_Companion': 'Group', 'Activity_Level': 5, 'Budget': 'Mid-range'},  # Should be Party or Adventure
    {'Age': 45, 'Gender': 'Male', 'Income_Level': 'High', 'Travel_Companion': 'Solo', 'Activity_Level': 3, 'Budget': 'Mid-range'},  # Should be Cultural
]

for i, inp in enumerate(test_inputs):
    input_data = pd.DataFrame([inp])

    # One-hot encode
    input_encoded = pd.get_dummies(input_data, columns=['Gender', 'Income_Level', 'Travel_Companion', 'Budget'], drop_first=True)

    # Scale numerical
    numerical_features = ['Age', 'Activity_Level']
    input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])

    # Ensure columns
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].iloc[0]

    # Predict
    prediction_encoded = model.predict(final_input)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = model.predict_proba(final_input)[0]

    print(f"Test {i+1}: {inp}")
    print(f"Predicted: {prediction_label}")
    print(f"Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")
    print()
