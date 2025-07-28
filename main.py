import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import time
import ssl


# --- 1. Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function. Maps to (0, 1)."""
    # Add a clip to avoid overflow for large negative numbers
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """Hyperbolic Tangent activation function. Maps to (-1, 1)."""
    return np.tanh(x)


def relu(x):
    """Rectified Linear Unit activation function. f(x) = max(0, x)."""
    return np.maximum(0, x)


def gaussian(x):
    """Gaussian (RBF) activation function."""
    return np.exp(-x ** 2)


# Dictionary to hold the functions for easy iteration
ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'gaussian': gaussian
}


# --- 2. Core ELM Functions ---

def calculate_output_weights(H, T_train, C=1e-5):
    """
    Calculate the output weights (beta) using the Moore-Penrose pseudoinverse
    with L2 regularization (Ridge Regression).
    """
    num_hidden_neurons = H.shape[1]
    identity = np.identity(num_hidden_neurons)
    # The core calculation: beta = (H^T * H + C*I)^-1 * H^T * T
    H_pseudo_inv = np.linalg.inv(H.T @ H + C * identity) @ H.T
    beta = H_pseudo_inv @ T_train
    return beta


# --- 3. Experiment 1: Standard ELM with Input Downsampling ---
def run_downsampling_experiment(x_train_full, T_train_full, x_test, y_test, T_test, activation_func,
                                num_hidden_neurons):
    """
    Simulates an ELM where the input image is first downsampled (e.g., via PCA),
    and then the entire resulting vector is mapped to a hidden layer.
    """
    n_components = 50
    pca = PCA(n_components=n_components)
    x_train_downsampled = pca.fit_transform(x_train_full)
    x_test_downsampled = pca.transform(x_test)

    input_size = n_components

    np.random.seed(42)
    input_weights = np.random.normal(size=[input_size, num_hidden_neurons])
    biases = np.random.normal(size=[num_hidden_neurons])

    H_train = activation_func(x_train_downsampled @ input_weights + biases)
    beta = calculate_output_weights(H_train, T_train_full)

    H_test = activation_func(x_test_downsampled @ input_weights + biases)
    T_pred = H_test @ beta
    y_pred = np.argmax(T_pred, axis=1)
    return accuracy_score(y_test, y_pred)


# --- 4. Experiment 2: ELM with Temporal Upsampling Simulation ---
def run_temporal_upsampling_experiment(x_train_full, T_train_full, x_test, y_test, T_test, activation_func,
                                       upsampling_factor):
    """
    Simulates the temporal upsampling concept. Each input feature (pixel) is
    individually expanded into a vector of new features.
    """
    n_pixels = 15
    pca = PCA(n_components=n_pixels)
    x_train_pixels = pca.fit_transform(x_train_full)
    x_test_pixels = pca.transform(x_test)

    np.random.seed(42)
    upsampling_weights = np.random.normal(size=[1, upsampling_factor])
    upsampling_biases = np.random.normal(size=[upsampling_factor])

    def temporal_upsampler(pixel_value_vector):
        p = pixel_value_vector[:, :, np.newaxis]
        expanded_features = activation_func(p * upsampling_weights + upsampling_biases)
        return expanded_features.reshape(pixel_value_vector.shape[0], -1)

    H_train = temporal_upsampler(x_train_pixels)
    beta = calculate_output_weights(H_train, T_train_full)

    H_test = temporal_upsampler(x_test_pixels)
    T_pred = H_test @ beta
    y_pred = np.argmax(T_pred, axis=1)
    return accuracy_score(y_test, y_pred)


# --- 5. Main Execution ---
if __name__ == '__main__':
    print("Loading and preparing data...")
    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    T_train_full = onehot_encoder.fit_transform(y_train_full.reshape(-1, 1))
    T_test = onehot_encoder.transform(y_test.reshape(-1, 1))
    print("Data ready.")

    # --- Define Experiment Hyperparameters ---
    neuron_counts = [500, 1000, 2000]  # For Downsampling ELM
    upsampling_factors = [32, 64, 128]  # For Temporal Upsampling ELM

    # --- Run Experiments ---
    for func_name, activation_func in ACTIVATION_FUNCTIONS.items():
        print("\n" + "=" * 25 + f" ACTIVATION FUNCTION: {func_name.upper()} " + "=" * 25)

        # --- Experiment 1: Downsampling ---
        print("\n----- Running Downsampling ELM Experiment -----")
        for n_neurons in neuron_counts:
            print(f"  Testing with {n_neurons} hidden neurons...")
            start_time = time.time()
            accuracy = run_downsampling_experiment(
                x_train_full, T_train_full, x_test, y_test, T_test,
                activation_func, n_neurons
            )
            end_time = time.time()
            print(f"    -> Accuracy: {accuracy * 100:.2f}%  (took {end_time - start_time:.2f}s)")

        # --- Experiment 2: Temporal Upsampling ---
        print("\n----- Running Temporal Upsampling ELM Experiment -----")
        for factor in upsampling_factors:
            n_pixels = 15
            final_features = n_pixels * factor
            print(f"  Testing with upsampling factor {factor} (Total features: {final_features})...")
            start_time = time.time()
            accuracy = run_temporal_upsampling_experiment(
                x_train_full, T_train_full, x_test, y_test, T_test,
                activation_func, factor
            )
            end_time = time.time()
            print(f"    -> Accuracy: {accuracy * 100:.2f}%  (took {end_time - start_time:.2f}s)")

    print("\n" + "=" * 70)
    print("All experiments complete.")
