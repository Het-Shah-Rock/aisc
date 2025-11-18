'''
AIM : To design and implement a Hopfield Network that can store and recall binary
representations of handwritten digits. The network should be able to reconstruct
the original digits when given noisy or partial versions of them.
Theory: The problem involves designing a Hopfield Network to store and recall binary
representations of handwritten digits from the MNIST dataset. A Hopfield
Network is a type of recurrent neural network that can act as an associative
memory, storing patterns and retrieving them when given noisy or incomplete
versions. The key idea is to train the network to memorize specific patterns,
then recall them when presented with noisy or corrupted inputs.
To solve this problem, we first need to preprocess the MNIST dataset,
converting the images of digits into binary form: pixels are represented as +1
(for the stroke) or -1 (for the background). We will select specific digits, such as
odd numbers or multiples of 3, and convert each digit image into a 1D vector.
Next, during the training phase, we apply the Hebbian learning rule to update
the weight matrix of the Hopfield Network. This rule strengthens the
connections between neurons that are simultaneously active. In the testing
phase, we simulate noisy inputs by randomly flipping some pixels, then present
the noisy pattern to the network. The network will iteratively update its states,
converging to a stored pattern.
Finally, we evaluate the network's performance by comparing the recalled digit
with the original stored one and calculating the recall accuracy. Python libraries
such as NumPy can be used to implement the Hopfield Network and handle the
image data.
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Choose specific digits to store (e.g., odd numbers 1, 3, 5, 7, 9)
selected_digits = [1, 3, 5, 7, 9]

# Filter the training data for the selected digits
train_indices = np.isin(y_train, selected_digits)
x_train_selected = x_train[train_indices]
y_train_selected = y_train[train_indices]

# Convert images to binary (-1 and +1)
def binarize_image(image):
    return np.where(image > 0, 1, -1)

# Binarize the selected images
binary_images = np.array([binarize_image(img) for img in x_train_selected])

# Flatten the images into 1D vectors
flattened_patterns = binary_images.reshape(binary_images.shape[0], -1)

# Select one example for each digit
unique_patterns = []
for digit in selected_digits:
    idx = np.where(y_train_selected == digit)[0][0]
    unique_patterns.append(flattened_patterns[idx])

unique_patterns = np.array(unique_patterns)

# Initialize the weight matrix
num_neurons = unique_patterns.shape[1]
W = np.zeros((num_neurons, num_neurons))

# Hebbian Learning Rule: W = sum(p_mu * p_mu^T) for all stored patterns p_mu
for pattern in unique_patterns:
    W += np.outer(pattern, pattern)

# Set diagonal elements to zero
np.fill_diagonal(W, 0)

# Function to add noise to a pattern
def add_noise(pattern, noise_level):
    noisy_pattern = np.copy(pattern)
    num_pixels = len(noisy_pattern)
    num_flips = int(num_pixels * noise_level)
    flip_indices = np.random.choice(num_pixels, num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

# Function to update the network
def update_network(pattern, W, max_iterations=100):
    current_pattern = np.copy(pattern)
    previous_state = np.zeros_like(current_pattern)

    for _ in range(max_iterations):
        previous_state = np.copy(current_pattern)
        for i in np.random.permutation(len(current_pattern)):
            activation = np.dot(W[i, :], current_pattern)
            current_pattern[i] = 1 if activation >= 0 else -1

        if np.array_equal(current_pattern, previous_state):
            break

    return current_pattern

# Test with a noisy version of one of the stored patterns (e.g., the digit 3)
test_pattern = unique_patterns[1]
noise_level = 0.25
noisy_input = add_noise(test_pattern, noise_level)
recalled_pattern = update_network(noisy_input, W)

# Reshape patterns back to 28x28 for visualization
original_image = test_pattern.reshape(28, 28)
noisy_image = noisy_input.reshape(28, 28)
recalled_image = recalled_pattern.reshape(28, 28)

# Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Stored Pattern')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title(f'Noisy Input ({int(noise_level * 100)}% noise)')
axes[1].axis('off')

axes[2].imshow(recalled_image, cmap='gray')
axes[2].set_title('Recalled Pattern')
axes[2].axis('off')

plt.show()

# Function to evaluate the recall accuracy
def evaluate_recall(stored_patterns, num_tests=100, noise_level=0.2):
    correct_recalls = 0
    total_tests = 0

    for _ in range(num_tests):
        original_pattern_idx = np.random.randint(len(stored_patterns))
        original_pattern = stored_patterns[original_pattern_idx]
        noisy_input = add_noise(original_pattern, noise_level)
        recalled_pattern = update_network(noisy_input, W)

        distances = [np.linalg.norm(recalled_pattern - stored_p) for stored_p in stored_patterns]
        closest_match_idx = np.argmin(distances)

        if closest_match_idx == original_pattern_idx:
            correct_recalls += 1

        total_tests += 1

    accuracy = (correct_recalls / total_tests) * 100
    print(f"\nRecall accuracy with {int(noise_level * 100)}% noise: {accuracy:.2f}%")
    return accuracy

# Run the evaluation with different noise levels
evaluate_recall(unique_patterns, num_tests=10, noise_level=0.1)
evaluate_recall(unique_patterns, num_tests=10, noise_level=0.2)
evaluate_recall(unique_patterns, num_tests=10, noise_level=0.3)
