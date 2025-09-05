# autoencoder_spectroscopy.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg


# Create folder structure

BASE_DIR = os.path.join(os.getcwd(), "autoencoder_spectroscopy")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

for d in [PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("All outputs will be saved under:", BASE_DIR)

# Generate synthetic spectroscopy data

def generate_synthetic_spectra(num_samples=500, num_points=128):
    
    #Generates synthetic spectral lines with random peaks and added Gaussian noise
    
    X_clean = np.zeros((num_samples, num_points))
    X_noisy = np.zeros_like(X_clean)
    
    for i in range(num_samples):
        # Random peaks
        n_peaks = np.random.randint(1,4)
        x = np.linspace(0, 1, num_points)
        signal = np.zeros(num_points)
        for _ in range(n_peaks):
            peak_center = np.random.uniform(0.2,0.8)
            width = np.random.uniform(0.02,0.08)
            amplitude = np.random.uniform(0.5,1.0)
            signal += amplitude * np.exp(-(x-peak_center)**2/(2*width**2))
        X_clean[i,:] = signal
        # Add Gaussian noise
        X_noisy[i,:] = signal + 0.1*np.random.randn(num_points)
    return X_noisy, X_clean

print("Generating synthetic spectroscopy signals...")
X_noisy, X_clean = generate_synthetic_spectra(num_samples=500, num_points=128)
X_test_noisy, X_test_clean = generate_synthetic_spectra(num_samples=50, num_points=128)

# Visualize sample signals
def plot_spectra_samples(X_noisy, X_clean, save_path=None):
    fig = plt.figure(figsize=(12,4))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.plot(X_clean[i], label="Clean")
        plt.plot(X_noisy[i], label="Noisy", alpha=0.6)
        plt.xticks([]); plt.yticks([])
        if i==0: plt.legend(fontsize='small')
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

plot_spectra_samples(X_noisy, X_clean, save_path=os.path.join(PLOTS_DIR,"sample_noisy_clean.png"))


# Build autoencoder model

input_dim = X_noisy.shape[1]
inp = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(inp)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inp, decoded)

# Save and display model plot
plot_model(autoencoder, to_file=os.path.join(PLOTS_DIR,'autoencoder_model.png'), show_shapes=True)
data = mpimg.imread(os.path.join(PLOTS_DIR,'autoencoder_model.png'))
plt.figure(figsize=(10,10))
plt.imshow(data)
plt.axis('off')
plt.title("Autoencoder Architecture")
plt.savefig(os.path.join(PLOTS_DIR,"autoencoder_model_display.png"), dpi=200)
plt.show()


# Compile and summarize

autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
autoencoder.summary()
with open(os.path.join(MODELS_DIR,"autoencoder_summary.txt"), "w", encoding="utf-8") as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + "\n"))

# Train autoencoder

history = autoencoder.fit(X_noisy, X_clean,
                          validation_data=(X_test_noisy,X_test_clean),
                          epochs=50,
                          batch_size=32,
                          verbose=2)

# Plot training loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR,"training_loss.png"), dpi=200)
plt.show()

# Evaluate and visualize reconstructions

X_denoised = autoencoder.predict(X_test_noisy)

fig = plt.figure(figsize=(12,6))
for i in range(5):
    plt.subplot(3,5,i+1)
    plt.plot(X_test_noisy[i]); plt.title("Noisy"); plt.xticks([]); plt.yticks([])
    plt.subplot(3,5,5+i+1)
    plt.plot(X_test_clean[i]); plt.title("Clean"); plt.xticks([]); plt.yticks([])
    plt.subplot(3,5,10+i+1)
    plt.plot(X_denoised[i]); plt.title("Denoised"); plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,"denoising_results.png"), dpi=200)
plt.show()

print("Autoencoder for spectroscopy denoising completed. All outputs saved under:", BASE_DIR)
