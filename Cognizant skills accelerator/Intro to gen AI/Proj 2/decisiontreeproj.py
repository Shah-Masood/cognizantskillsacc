import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset from CSV
file_path = "weather_data.csv"  # Ensure the file is in the correct directory
weather_df = pd.read_csv(file_path)

# Convert precipitation into categorical values (0 = No Rain, 1 = Light Rain, 2 = Heavy Rain)
def categorize_precipitation(value):
    if value == 0:
        return 0  # No Rain
    elif value <= 10:
        return 1  # Light Rain
    else:
        return 2  # Heavy Rain

weather_df["Precipitation Category"] = weather_df["Precipitation (mm)"].apply(categorize_precipitation)

# Select features and target variable
X = weather_df[["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)"]]
y = weather_df["Precipitation Category"]

# Normalize the features
X = (X - X.min()) / (X.max() - X.min())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5, min_samples_split=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["No Rain", "Light Rain", "Heavy Rain"])

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Show confusion matrix as a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=["No Rain", "Light Rain", "Heavy Rain"], 
                              columns=["No Rain", "Light Rain", "Heavy Rain"])
print("\nConfusion Matrix DataFrame:")
print(conf_matrix_df)
