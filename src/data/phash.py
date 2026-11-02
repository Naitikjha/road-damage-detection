"""
Perceptual Hashing (dHash) and Near-Duplicate Detection.

Why dHash for Road Damage Datasets:
RDD2022 captures sequential dashcam frames. When a vehicle waits at a stoplight
or drives slowly, dozens of nearly identical frames are recorded.
If split randomly, near-identical frames would leak across train, tune, and val
splits, giving falsely high validation scores.

dHash (Difference Hash) works by:
1. Converting the image to grayscale.
2. Resizing to (hash_size + 1, hash_size), e.g., 9x8 for a 64-bit hash.
3. Comparing adjacent horizontal pixels: pixel[col+1] > pixel[col].
4. Calculating the Hamming distance between hashes. A distance <= max_dist
   identifies near-duplicate frames.
"""

from typing import Union, Tuple
import cv2
import numpy as np
from PIL import Image


def compute_dhash(image_input: Union[str, np.ndarray, Image.Image], hash_size: int = 8) -> np.ndarray:
    """
    Compute difference hash (dHash) as a boolean NumPy array of shape (hash_size * hash_size,).

    Args:
        image_input: File path (str), BGR/RGB NumPy array, or PIL Image.
        hash_size: Width/Height of the hash grid (default 8 -> 64-bit hash).

    Returns:
        np.ndarray: 1D boolean array representing the hash bits.
    """
    if isinstance(image_input, str):
        # Read with OpenCV for speed
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_input}")
    elif isinstance(image_input, Image.Image):
        gray = image_input.convert("L")
        img = np.array(gray)
    elif isinstance(image_input, np.ndarray):
        if img_is_color := (len(image_input.shape) == 3):
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize to (width = hash_size + 1, height = hash_size)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    # Compare adjacent pixels along each row: col[1:] > col[:-1]
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """
    Calculate the Hamming distance (number of mismatched bits) between two dHashes.

    Args:
        hash1: 1D boolean array
        hash2: 1D boolean array

    Returns:
        int: Number of differing bits.
    """
    return int(np.count_nonzero(hash1 != hash2))


def hash_to_hex(hash_arr: np.ndarray) -> str:
    """Convert boolean hash array into a human-readable hex string."""
    packed = np.packbits(hash_arr)
    return packed.tobytes().hex()


def hex_to_hash(hex_str: str, hash_size: int = 8) -> np.ndarray:
    """Convert hex string back to a boolean hash array."""
    raw_bytes = bytes.fromhex(hex_str)
    unpacked = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    return unpacked[: hash_size * hash_size].astype(bool)
