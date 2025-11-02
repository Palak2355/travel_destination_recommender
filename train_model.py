import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import os

def generate_synthetic_travel_data(num_rows=5000):
    """
    Generates a synthetic dataset for the Travel Recommendation System.
    The logic ensures a reasonable correlation between features and the target destination.
    """
    np.random.seed(42)

    # Feature choices
    ages = np.random.randint(18, 65, num_rows)
    genders = np.random.choice(['Male', 'Female', 'Other'], num_rows, p=[0.45, 0.45, 0.1])
    incomes = np.random.choice(['Low', 'Medium', 'High'], num_rows, p=[0.3, 0.4, 0.3])
    companions = np.random.choice(['Solo', 'Family', 'Group'], num_rows, p=[0.35, 0.4, 0.25])
    activity_levels = np.random.randint(1, 6, num_rows)
    budgets = np.random.choice(['Economical', 'Mid-range', 'Luxury'], num_rows, p=[0.3, 0.4, 0.3])

    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Income_Level': incomes,
        'Travel_Companion': companions,
        'Activity_Level': activity_levels,
        'Budget': budgets,
    })

    # Map destination with refined logic to reduce bias towards Cultural/Historical
    def map_destination(row):
        # Adventure/Nature: High activity, younger, solo/group, mid/low budget
        if (row['Activity_Level'] >= 4) and (row['Age'] < 40) and (row['Budget'] in ['Economical', 'Mid-range']):
            return 'Adventure/Nature'

        # Relaxation/Beach: Low activity, family/solo, high budget, older
        elif (row['Activity_Level'] <= 2) and (row['Budget'] in ['Luxury', 'Mid-range']):
            return 'Relaxation/Beach'

        # Party/Nightlife: High activity and young
        elif (row['Activity_Level'] >= 4) and (row['Age'] < 35):
            return 'Party/Nightlife'

        # Cultural/Historical: Mid activity, older age, specific budget and companion
        elif (row['Activity_Level'] in [2, 3]) and (row['Age'] >= 45) and (row['Budget'] in ['Mid-range', 'Luxury']):
            return 'Cultural/Historical'

        # Fallback to random choice for balance
        return np.random.choice(['Adventure/Nature', 'Relaxation/Beach', 'Cultural/Historical', 'Party/Nightlife'])

    df['Target_Destination'] = df.apply(map_destination, axis=1)

    # Add some noise/overlap
    num_noise = int(num_rows * 0.1)  # Increased noise to 10% for better balance
    noise_indices = np.random.choice(df.index, num_noise, replace=False)

    destination_options = ['Adventure/Nature', 'Relaxation/Beach', 'Cultural/Historical', 'Party/Nightlife']

    for i in noise_indices:
        current_dest = df.loc[i, 'Target_Destination']
        new_dest = np.random.choice([d for d in destination_options if d != current_dest])
        df.loc[i, 'Target_Destination'] = new_dest

    return df

# Generate data
df = generate_synthetic_travel_data()

# Save data
df.to_csv('travel_data.csv', index=False)

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

# Train model with tuned parameters
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")

# Save model assets
dump(model, 'best_rf_model.joblib')
dump(scaler, 'scaler.joblib')
dump(X_train.columns.tolist(), 'feature_columns.joblib')
dump(le, 'label_encoder.joblib')

print("Model assets saved.")
