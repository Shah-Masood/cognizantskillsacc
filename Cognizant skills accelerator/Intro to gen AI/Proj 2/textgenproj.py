import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Load a sample text corpus (Shakespeare's works)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path_to_file = keras.utils.get_file("shakespeare.txt", url)

# Read text
with open(path_to_file, "r", encoding="utf-8") as file:
    text = file.read().lower()  # Convert to lowercase

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
words = text.split()

SEQUENCE_LENGTH = 50  # Each sequence will have 50 words
for i in range(len(words) - SEQUENCE_LENGTH):
    input_sequences.append(tokenizer.texts_to_sequences([" ".join(words[i:i+SEQUENCE_LENGTH+1])])[0])

# Convert sequences to numpy array
input_sequences = np.array(input_sequences)

# Split input (X) and labels (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)  # One-hot encode labels

# Build LSTM model
model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=SEQUENCE_LENGTH),
    keras.layers.LSTM(150, return_sequences=True),
    keras.layers.LSTM(150),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(total_words, activation="softmax")  # Predict next word
])

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
history = model.fit(X, y, epochs=30, batch_size=128, validation_split=0.2)

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("LSTM Training Performance")
plt.legend()
plt.show()

# Function to generate text
def generate_text(seed_text, next_words=20, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=SEQUENCE_LENGTH, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs ** (1/temperature))
        predicted_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + predicted_word

    return seed_text

# Generate text from a seed sentence
seed_sentence = "to be or not to be"
generated_text = generate_text(seed_sentence, next_words=30, temperature=0.8)

print("\nGenerated Text:\n", generated_text)
