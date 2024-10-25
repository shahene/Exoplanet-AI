import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Replace 'path_to_koi.csv' with the actual path where you saved the KOI table CSV
file_path = './kepler_cumulative.csv'

# Load the CSV file into a pandas DataFrame
df_koi = pd.read_csv(file_path)

# Fill missing values in numeric columns with the median
numeric_columns = df_koi.select_dtypes(include=['float', 'int']).columns
df_koi[numeric_columns] = df_koi[numeric_columns].fillna(df_koi[numeric_columns].median())

# Select relevant features
features = ['koi_period', 'koi_prad', 'koi_depth', 'koi_impact']
X = df_koi[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the target variable (labels)
y = df_koi['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy of the Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Model Accuracy:", rf_accuracy)
