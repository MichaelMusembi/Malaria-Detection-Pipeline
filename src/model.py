"""
Model Training Module for Malaria Detection Pipeline

Creates, trains, and saves MobileNetV2-based transfer learning model.
"""

import os
import json
from datetime import datetime
from typing import Tuple, Optional, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam

from preprocessing import (
    create_train_generator,
    create_validation_generator,
    calculate_class_weights,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE
)


def create_model(
    input_shape: Tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, 3),
    learning_rate: float = 0.0001,
    fine_tune_layers: int = 50
) -> keras.Model:
    """
    Creates a MobileNetV2-based transfer learning model for binary classification.
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        learning_rate: Initial learning rate for Adam optimizer
        fine_tune_layers: Number of top layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation layer (applied during training only)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    
    # Preprocessing for MobileNetV2
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def unfreeze_model(
    model: keras.Model,
    num_layers: int = 50
) -> keras.Model:
    """
    Unfreezes the top layers of the base model for fine-tuning.
    
    Args:
        model: Compiled Keras model
        num_layers: Number of layers from the top to unfreeze
        
    Returns:
        Model with unfrozen layers
    """
    # Get the base model (MobileNetV2)
    base_model = model.layers[4]  # Adjust index if architecture changes
    
    # Unfreeze the top layers
    base_model.trainable = True
    
    # Freeze all layers except the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_callbacks(
    model_path: str = 'models/malaria_mobilenet.h5',
    patience: int = 5
) -> list:
    """
    Creates training callbacks for model optimization.
    
    Args:
        model_path: Path to save the best model
        patience: Number of epochs to wait before early stopping
        
    Returns:
        List of Keras callbacks
    """
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(
    train_dir: str,
    epochs: int = 20,
    batch_size: int = BATCH_SIZE,
    model_path: str = 'models/malaria_mobilenet.h5',
    fine_tune: bool = True
) -> Tuple[keras.Model, dict]:
    """
    Trains the malaria detection model with transfer learning.
    
    Args:
        train_dir: Path to training data directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_path: Path to save the trained model
        fine_tune: Whether to fine-tune the base model
        
    Returns:
        Tuple of (trained model, training history)
    """
    print("🔧 Creating data generators...")
    train_gen = create_train_generator(train_dir, batch_size=batch_size)
    val_gen = create_validation_generator(train_dir, batch_size=batch_size)
    
    print(f"📊 Training samples: {train_gen.samples}")
    print(f"📊 Validation samples: {val_gen.samples}")
    print(f"📊 Classes: {train_gen.class_indices}")
    
    # Calculate class weights
    print("⚖️ Calculating class weights...")
    class_weights = calculate_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("🏗️ Building model...")
    model = create_model()
    print(model.summary())
    
    # Initial training (frozen base)
    print(f"\n🚀 Starting initial training for {epochs} epochs...")
    callbacks = get_callbacks(model_path=model_path)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Fine-tuning (optional)
    if fine_tune:
        print("\n🔥 Fine-tuning model...")
        model = unfreeze_model(model, num_layers=50)
        
        fine_tune_epochs = 10
        total_epochs = epochs + fine_tune_epochs
        
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Merge histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    # Save final model
    print(f"\n💾 Saving final model to {model_path}")
    model.save(model_path)
    
    # Save training history
    history_path = model_path.replace('.h5', '_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types
        history_dict = {
            k: [float(v) for v in vals] 
            for k, vals in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    
    # Save model version info
    version_info = {
        "version": "1.0.0",
        "trained_date": datetime.now().isoformat(),
        "epochs": len(history.history['loss']),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "model_path": model_path
    }
    
    version_path = 'models/model_version.json'
    with open(version_path, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    print("✅ Training complete!")
    print(f"📈 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"📉 Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    return model, history.history


def load_model_from_path(model_path: str = 'models/malaria_mobilenet.h5') -> keras.Model:
    """
    Loads a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📥 Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    
    return model


def evaluate_model(
    model: keras.Model,
    test_dir: str,
    batch_size: int = BATCH_SIZE
) -> Dict[str, float]:
    """
    Evaluates the model on test data.
    
    Args:
        model: Trained Keras model
        test_dir: Path to test data directory
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    from preprocessing import create_test_generator
    
    print("📊 Creating test generator...")
    test_gen = create_test_generator(test_dir, batch_size=batch_size)
    
    print("🧪 Evaluating model...")
    results = model.evaluate(test_gen, verbose=1)
    
    metrics = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'test_precision': float(results[2]),
        'test_recall': float(results[3]),
        'test_auc': float(results[4])
    }
    
    # Calculate F1 score
    precision = metrics['test_precision']
    recall = metrics['test_recall']
    if precision + recall > 0:
        metrics['test_f1'] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics['test_f1'] = 0.0
    
    print("\n📊 Test Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    """
    Train the model when script is run directly
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train malaria detection model')
    parser.add_argument('--train-dir', type=str, default='data/train',
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--no-fine-tune', action='store_true',
                       help='Skip fine-tuning phase')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=not args.no_fine_tune
    )
    
    print("\n✅ Model training complete!")