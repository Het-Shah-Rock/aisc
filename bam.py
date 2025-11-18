'''
AIM : To design and implement a Bidirectional Associative Memory (BAM) network
that learns associations between English words and their Hindi translations. The
BAM should recall the correct translation in both directions (Eng → Hindi,
Hindi → Eng).
Theory: The Bidirectional Associative Memory (BAM) is a type of recurrent neural
network that uses a symmetric weight matrix to establish bidirectional
associations between two sets of data. In the context of this project, the BAM
network is designed to learn the translation associations between English and
Hindi words. The main objective is to build a system that can recall the correct
translation in both directions: from English to Hindi and vice versa.
To begin, we define a bilingual dictionary with pairs of English and Hindi
words, such as “dog” → “कुत्ता” and “cat” → “बिल्ली”. These words are then
encoded into binary vectors using a one-hot encoding scheme, where each
character in a word is mapped to a unique index, and the word is represented as
a vector of binary values.
The BAM network consists of two layers: an input layer for the English words
(in binary) and an output layer for the corresponding Hindi translations. The
weight matrix, initialized with zeros, is updated using the Hebbian learning rule,
which reinforces the connections between word pairs. After training, the
network is tested by inputting a word in one language and checking if it recalls
the correct translation in the other language.
Finally, the accuracy of the system is evaluated by testing its robustness and
ability to recall correct translations, even with noisy input.
'''

import numpy as np
import matplotlib.pyplot as plt

dictionary = {
    'dog': 'कुत्ता',
    'cat': 'बिल्ली',
    'sun': 'सूरज',
    'moon': 'चाँद'
}

english_chars = sorted(list(set(''.join(dictionary.keys()))))
hindi_chars = sorted(list(set(''.join(dictionary.values()))))

eng_char_to_idx = {char: i for i, char in enumerate(english_chars)}
hin_char_to_idx = {char: i for i, char in enumerate(hindi_chars)}

max_eng_len = max(len(word) for word in dictionary.keys())
max_hin_len = max(len(word) for word in dictionary.values())

eng_vector_dim = len(english_chars) * max_eng_len
hin_vector_dim = len(hindi_chars) * max_hin_len

def encode_word(word, char_to_idx, max_len, vector_dim):
    vector = np.full(vector_dim, -1)
    char_len = len(char_to_idx)
    for i, char in enumerate(word):
        if char in char_to_idx:
            char_idx = char_to_idx[char]
            start_idx = i * char_len + char_idx
            if start_idx < vector_dim:
                vector[start_idx] = 1
    return vector

encoded_pairs = []
for eng_word, hin_word in dictionary.items():
    X = encode_word(eng_word, eng_char_to_idx, max_eng_len, eng_vector_dim)
    Y = encode_word(hin_word, hin_char_to_idx, max_hin_len, hin_vector_dim)
    encoded_pairs.append((X, Y))

W = np.zeros((eng_vector_dim, hin_vector_dim))

for X, Y in encoded_pairs:
    X_reshaped = X.reshape(-1, 1)
    Y_reshaped = Y.reshape(1, -1)
    W += np.dot(X_reshaped, Y_reshaped)

def decode_word(vector, char_to_idx, max_len):
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    decoded_word = ''
    char_len = len(idx_to_char)
    vector_2d = vector.reshape(max_len, char_len)

    for i in range(max_len):
        char_vector = vector_2d[i]
        try:
            char_idx = np.where(char_vector == 1)[0][0]
            if char_idx in idx_to_char:
                decoded_word += idx_to_char[char_idx]
        except IndexError:
            pass
    return decoded_word

def recall(input_vector, W, direction='forward', iterations=5):
    if direction == 'forward':
        X = input_vector.copy()
        Y = np.zeros(W.shape[1])
    elif direction == 'reverse':
        Y = input_vector.copy()
        X = np.zeros(W.shape[0])
    else:
        raise ValueError("Direction must be 'forward' or 'reverse'")

    for _ in range(iterations):
        Y_new = np.dot(X, W)
        Y_new = np.where(Y_new >= 0, 1, -1)

        X_new = np.dot(Y_new, W.T)
        X_new = np.where(X_new >= 0, 1, -1)

        if np.array_equal(X_new, X) and np.array_equal(Y_new, Y):
            break

        X = X_new
        Y = Y_new

    return X, Y

print("\n--- Testing: English to Hindi ---")
test_eng_word = 'dog'
test_eng_vector = encode_word(test_eng_word, eng_char_to_idx, max_eng_len, eng_vector_dim)

# Adding noise: Flip a few bits
noisy_input = test_eng_vector.copy()
noise_indices = np.random.choice(eng_vector_dim, 10, replace=False)
noisy_input[noise_indices] *= -1

recalled_eng_vector, recalled_hin_vector = recall(noisy_input, W)
recalled_hin_word = decode_word(recalled_hin_vector, hin_char_to_idx, max_hin_len)

print(f"Input English word (noisy): {test_eng_word}")
print(f"Recalled Hindi word: {recalled_hin_word}")
print(f"Correct Translation: {dictionary[test_eng_word]}")

print("\n--- Testing: Hindi to English ---")
test_hin_word = 'सूरज'
test_hin_vector = next((Y for X, Y in encoded_pairs if decode_word(X, eng_char_to_idx, max_eng_len) == 'sun'), None)

if test_hin_vector is None:
    print("Error: Hindi word not found in encoded pairs.")
else:
    recalled_eng_vector, recalled_hin_vector = recall(test_hin_vector, W, direction='reverse')
    recalled_eng_word = decode_word(recalled_eng_vector, eng_char_to_idx, max_eng_len)

    correct_eng_word = next((key for key, value in dictionary.items() if value == test_hin_word), None)

    print(f"Input Hindi word: {test_hin_word}")
    print(f"Recalled English word: {recalled_eng_word}")
    print(f"Correct Translation: {correct_eng_word}")

def evaluate_bam_accuracy(pairs, W, noise_level=0.1):
    total_tests = len(pairs)
    correct_forward = 0
    correct_reverse = 0

    print(f"\n--- Evaluation with {int(noise_level * 100)}% noise ---")
    for X, Y in pairs:
        noisy_X = X.copy()
        if noise_level > 0:
            noise_indices = np.random.choice(len(X), int(len(X) * noise_level), replace=False)
            noisy_X[noise_indices] *= -1

        _, recalled_Y = recall(noisy_X, W, direction='forward')
        if np.array_equal(recalled_Y, Y):
            correct_forward += 1

        noisy_Y = Y.copy()
        if noise_level > 0:
            noise_indices = np.random.choice(len(Y), int(len(Y) * noise_level), replace=False)
            noisy_Y[noise_indices] *= -1

        recalled_X, _ = recall(noisy_Y, W, direction='reverse')
        if np.array_equal(recalled_X, X):
            correct_reverse += 1

    forward_accuracy = (correct_forward / total_tests) * 100
    reverse_accuracy = (correct_reverse / total_tests) * 100

    print(f"Forward Accuracy (Eng -> Hin): {forward_accuracy:.2f}%")
    print(f"Reverse Accuracy (Hin -> Eng): {reverse_accuracy:.2f}%")
    return forward_accuracy, reverse_accuracy

evaluate_bam_accuracy(encoded_pairs, W, noise_level=0.0)
evaluate_bam_accuracy(encoded_pairs, W, noise_level=0.1)
