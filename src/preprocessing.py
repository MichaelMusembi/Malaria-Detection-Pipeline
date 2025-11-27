"""
Image Preprocessing Module for Malaria Detection Pipeline

Handles image loading, augmentation, normalization, and data generator creation.
"""

import numpy as np
from typing import Tuple, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf


# Image configuration
IMG_HEIGHT: int = 128
IMG_WIDTH: int = 128
IMG_CHANNELS: int = 3
BATCH_SIZE: int = 32


def create_train_generator(
    train_dir: str,
    batch_size: int = BATCH_SIZE,
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    augment: bool = True
) -> ImageDataGenerator:
    """
    Creates an image data generator for training with augmentation.
    
    Args:
        train_dir: Path to training data directory
        batch_size: Number of images per batch
        target_size: Target image dimensions (height, width)
        augment: Whether to apply data augmentation
        
    Returns:
        Configured ImageDataGenerator for training
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=0.2,
            preprocessing_function=None
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=0.2
        )
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    return train_generator


def create_validation_generator(
    train_dir: str,
    batch_size: int = BATCH_SIZE,
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> ImageDataGenerator:
    """
    Creates an image data generator for validation (no augmentation).
    
    Args:
        train_dir: Path to training data directory (uses validation_split)
        batch_size: Number of images per batch
        target_size: Target image dimensions (height, width)
        
    Returns:
        Configured ImageDataGenerator for validation
    """
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )
    
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return validation_generator


def create_test_generator(
    test_dir: str,
    batch_size: int = BATCH_SIZE,
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> ImageDataGenerator:
    """
    Creates an image data generator for testing (no augmentation, no shuffle).
    
    Args:
        test_dir: Path to test data directory
        batch_size: Number of images per batch
        target_size: Target image dimensions (height, width)
        
    Returns:
        Configured ImageDataGenerator for testing
    """
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator


def preprocess_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> np.ndarray:
    """
    Preprocesses a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target image dimensions (height, width)
        
    Returns:
        Preprocessed image array with shape (1, height, width, 3)
    """
    # Load image
    img = load_img(image_path, target_size=target_size)
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_image_from_bytes(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> np.ndarray:
    """
    Preprocesses an image from bytes (for API uploads).
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target image dimensions (height, width)
        
    Returns:
        Preprocessed image array with shape (1, height, width, 3)
    """
    # Decode image from bytes
    img = tf.image.decode_image(image_bytes, channels=3)
    
    # Resize
    img = tf.image.resize(img, target_size)
    
    # Convert to numpy and normalize
    img_array = img.numpy() / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_class_names() -> Tuple[str, str]:
    """
    Returns the class names for the binary classification task.
    
    Returns:
        Tuple of (negative_class, positive_class)
    """
    return ("Uninfected", "Parasitized")


def calculate_class_weights(train_generator) -> dict:
    """
    Calculates class weights for handling imbalanced datasets.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get all labels
    labels = train_generator.classes
    
    # Calculate weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return {i: weight for i, weight in enumerate(class_weights)}


def augment_image(
    image: np.ndarray,
    augmentation_factor: int = 5
) -> list:
    """
    Applies random augmentation to create multiple versions of an image.
    
    Args:
        image: Input image array
        augmentation_factor: Number of augmented versions to create
        
    Returns:
        List of augmented image arrays
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Reshape if needed
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    augmented_images = []
    
    for _ in range(augmentation_factor):
        aug_iter = datagen.flow(image, batch_size=1)
        aug_image = next(aug_iter)[0]
        augmented_images.append(aug_image)
    
    return augmented_images


if __name__ == "__main__":
    """
    Test preprocessing functions
    """
    print("Testing preprocessing module...")
    
    # Test single image preprocessing
    test_image = np.random.rand(224, 224, 3)
    processed = preprocess_single_image.__wrapped__(test_image)
    print(f"Processed image shape: {processed.shape}")
    
    # Test augmentation
    augmented = augment_image(test_image, augmentation_factor=3)
    print(f"Generated {len(augmented)} augmented images")
    
    # Test class names
    classes = get_class_names()
    print(f"Class names: {classes}")
    
    print("✅ Preprocessing module tests passed!")