# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Load dataset
data = pd.read_csv('Telco-Customer-Churn.csv')

# Step 2: Preprocess the data

# Drop customerID as it is an identifier and not useful for prediction
data = data.drop('customerID', axis=1)

# Convert 'TotalCharges' to numeric (coerce errors will convert empty strings to NaN)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median (if any)
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Encode the target variable 'Churn' (Yes=1, No=0)
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# Identify categorical columns
cat_cols = data.select_dtypes(include=['object']).columns

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Step 3: Split data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 4: Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize features (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the Neural Network model

model = Sequential()
# Input layer + 1st hidden layer with 16 neurons, sigmoid activation (binary sigmoidal)
model.add(Dense(16, input_dim=X_train.shape[1], activation='sigmoid'))
# Optional: Add a second hidden layer (8 neurons)
model.add(Dense(8, activation='sigmoid'))
# Output layer with 1 neuron and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Step 8: Evaluate the model on test data
y_pred_prob = model.predict(X_test).flatten()
# Convert probabilities to binary predictions (threshold 0.5)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
