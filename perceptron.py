# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

data = pd.read_csv("/content/fruits_dataset.csv")

X = data[["Shape", "Texture", "Weight"]].values
y = data["Label"].values

# Split into train and test sets, and compile the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(3,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=SGD(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the model and store training history
history = model.fit(X_train, y_train, epochs=50, verbose=1)

# Evaluating model on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Predict on test samples
test_samples = np.array([
    [1, 1, 1],  # Apple
    [0, 0, 1],  # Orange
    [1, 0, 0],  # Apple
    [0, 1, 0],  # Orange
])
predictions = model.predict(test_samples)

print("\nPredictions on test samples:")
for features, pred_prob in zip(test_samples, predictions):
    fruit = "Apple" if pred_prob >= 0.5 else "Orange"
    print(f"Features: {features} → Prediction: {fruit} ({pred_prob[0]:.2f})")



# 1. Plot training loss over epochs
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 2. Plot training accuracy over epochs
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# 3. Bar graph for prediction probabilities on test samples
plt.subplot(1, 3, 3)
sample_labels = ['[1 1 1]', '[0 0 1]', '[1 0 0]', '[0 1 0]']
apple_probs = predictions.flatten()

bars = plt.bar(sample_labels, apple_probs, color=['green' if p >= 0.5 else 'orange' for p in apple_probs])
plt.title('Fruit Classification Predictions')
plt.ylabel('Probability of Apple')
plt.ylim(0, 1)
plt.grid(axis='y')

for bar, prob in zip(bars, apple_probs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'{prob:.2f}', ha='center')

plt.tight_layout()
plt.show()
