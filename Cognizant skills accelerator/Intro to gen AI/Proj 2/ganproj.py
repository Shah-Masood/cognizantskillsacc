# Import dataset using kagglehub
import kagglehub

# Download the Pistachio Image Dataset
dataset_path = kagglehub.dataset_download("muratkokludataset/pistachio-image-dataset")
print("Path to dataset files:", dataset_path)

# Proceeding with dataset processing and GAN training
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob

# Load images from dataset path
image_paths = glob(os.path.join(dataset_path, "*/*.jpg"))  # Adjust extension if necessary

# Preprocess images: Resize & Normalize
IMG_SIZE = 64  # Resize to 64x64
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))  # Resize
    img = img_to_array(img) / 255.0  # Normalize (0-1)
    return img

# Convert dataset into numpy array
images = np.array([preprocess_image(img_path) for img_path in image_paths])

# Check dataset shape
print(f"Loaded {len(images)} images with shape: {images.shape}")

# Define GAN Parameters
LATENT_DIM = 100  # Random noise size
CHANNELS = 3  # RGB

# Build Generator Model
def build_generator():
    model = keras.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')  # Output
    ])
    return model

# Build Discriminator Model
def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Output: Probability of being real
    ])
    return model

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# GAN Model (Generator + Discriminator)
discriminator.trainable = False  # Freeze discriminator for combined model

gan_input = keras.Input(shape=(LATENT_DIM,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

# Training Parameters
EPOCHS = 5000
BATCH_SIZE = 32

# Train GAN
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    # Train Discriminator
    idx = np.random.randint(0, images.shape[0], BATCH_SIZE)
    real_images = images[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, real_labels)

    # Print training progress
    if epoch % 500 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

# Generate new images
noise = np.random.normal(0, 1, (5, LATENT_DIM))
generated_images = generator.predict(noise)

# Plot generated images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow((generated_images[i] * 0.5) + 0.5)  # Rescale to [0,1] for display
    plt.axis("off")

plt.show()
