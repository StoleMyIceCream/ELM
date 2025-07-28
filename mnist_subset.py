import json
import requests
import random
import sys
import gzip
import io

# --- Configuration ---
# URL for the full MNIST training data in a gzipped JSON format
FULL_DATASET_URL = "https://github.com/lorenmh/mnist_handwritten_json/raw/refs/heads/master/mnist_handwritten_test.json.gz"

def create_mnist_subset():
    """
    Downloads the full, gzipped MNIST JSON dataset, decompresses it,
    creates a smaller random subset, and saves it in the format
    required by the interactive web app.
    """
    print(f"Attempting to download the gzipped MNIST dataset from:\n{FULL_DATASET_URL}\n")
    # Desired number of samples for our small, fast-loading subset
    NUM_TRAIN_SAMPLES = 9000
    NUM_TEST_SAMPLES = 1000

    # Output filename
    OUTPUT_FILENAME = "mnist_subset.json"

    # --- Step 1: Download and decompress the full dataset ---
    try:
        response = requests.get(FULL_DATASET_URL, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print("Downloading... (This may take a moment for the compressed file)")
        # Decompress the gzipped content in memory
        compressed_file = io.BytesIO(response.content)
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)

        # Load the JSON data from the decompressed content
        full_data = json.load(decompressed_file)
        print("Download and decompression complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download the dataset. {e}", file=sys.stderr)
        return
    except gzip.BadGzipFile:
        print("Error: The downloaded file is not a valid .gz file.", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON from the decompressed data.", file=sys.stderr)
        return

    if not isinstance(full_data, list):
        print("Error: Downloaded data is not in the expected list format.", file=sys.stderr)
        return

    print(f"Successfully loaded {len(full_data)} total samples.")

    # --- Step 2: Shuffle and create subsets ---
    print(f"Shuffling data and selecting {NUM_TRAIN_SAMPLES} for training and {NUM_TEST_SAMPLES} for testing...")
    random.shuffle(full_data)

    total_needed = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
    if len(full_data) < total_needed:
        print(
            f"Warning: The full dataset has fewer samples ({len(full_data)}) than requested ({total_needed}). Using all available data.",
            file=sys.stderr)
        NUM_TRAIN_SAMPLES = int(len(full_data) * 0.8)  # Adjust to 80/20 split
        NUM_TEST_SAMPLES = len(full_data) - NUM_TRAIN_SAMPLES

    # Split the shuffled data
    train_subset = full_data[:NUM_TRAIN_SAMPLES]
    test_subset = full_data[NUM_TRAIN_SAMPLES:total_needed]

    print("Subsets created.")

    # --- Step 3: Format the data for the web app ---
    # The web app expects separate lists for images and labels.
    final_structure = {
        "train_images": [item['image'] for item in train_subset],
        "train_labels": [item['label'] for item in train_subset],
        "test_images": [item['image'] for item in test_subset],
        "test_labels": [item['label'] for item in test_subset],
    }
    print("Data formatted into the final structure.")

    # --- Step 4: Save the new subset to a local file ---
    try:
        with open(OUTPUT_FILENAME, 'w') as f:
            json.dump(final_structure, f)
        print(f"\nSuccess! Your data subset has been saved as '{OUTPUT_FILENAME}'.")
        print("You can now upload this file to your GitHub repository along with 'index.html'.")

    except IOError as e:
        print(f"Error: Could not write to file '{OUTPUT_FILENAME}'. {e}", file=sys.stderr)


if __name__ == "__main__":
    create_mnist_subset()
