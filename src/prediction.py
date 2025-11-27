"""
Prediction Service Module for Malaria Detection Pipeline

Handles model loading, caching, and inference for single images.
"""

import os
import json
import time
from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocessing import (
    preprocess_single_image,
    preprocess_image_from_bytes,
    get_class_names,
    IMG_HEIGHT,
    IMG_WIDTH
)


class MalariaPredictionService:
    """
    Singleton service for malaria parasite detection predictions.
    Implements lazy loading and model caching.
    """
    
    _instance = None
    _model: Optional[keras.Model] = None
    _model_path: str = 'models/malaria_mobilenet.keras'
    _model_loaded: bool = False
    _class_names: Tuple[str, str] = get_class_names()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(MalariaPredictionService, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Loads the trained model from disk.
        
        Args:
            model_path: Optional custom path to model file
        """
        if model_path:
            self._model_path = model_path
        
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}. "
                "Please train the model first using src/model.py"
            )
        
        print(f"📥 Loading model from {self._model_path}...")
        start_time = time.time()
        
        self._model = keras.models.load_model(self._model_path)
        self._model_loaded = True
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
    
    def get_model(self) -> keras.Model:
        """
        Returns the loaded model, loading it if necessary (lazy loading).
        
        Returns:
            Loaded Keras model
        """
        if not self._model_loaded or self._model is None:
            self.load_model()
        return self._model
    
    def is_model_loaded(self) -> bool:
        """
        Checks if model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded and self._model is not None
    
    def reload_model(self) -> None:
        """
        Forces a model reload (useful after retraining).
        """
        print("🔄 Reloading model...")
        self._model = None
        self._model_loaded = False
        self.load_model()
    
    def predict_from_path(
        self,
        image_path: str,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Makes a prediction from an image file path.
        
        Args:
            image_path: Path to the image file
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Dictionary containing prediction results
        """
        # Ensure model is loaded
        model = self.get_model()
        
        # Preprocess image
        start_time = time.time()
        img_array = preprocess_single_image(image_path)
        preprocess_time = (time.time() - start_time) * 1000
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(img_array, verbose=0)
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        result = self._parse_prediction(
            prediction[0][0],
            threshold,
            preprocess_time + inference_time
        )
        
        return result
    
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Makes a prediction from raw image bytes (for API uploads).
        
        Args:
            image_bytes: Raw image bytes
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Dictionary containing prediction results
        """
        # Ensure model is loaded
        model = self.get_model()
        
        # Preprocess image
        start_time = time.time()
        img_array = preprocess_image_from_bytes(image_bytes)
        preprocess_time = (time.time() - start_time) * 1000
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(img_array, verbose=0)
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        result = self._parse_prediction(
            prediction[0][0],
            threshold,
            preprocess_time + inference_time
        )
        
        return result
    
    def _parse_prediction(
        self,
        score: float,
        threshold: float,
        processing_time: float
    ) -> Dict[str, any]:
        """
        Parses raw prediction score into structured result.
        
        Args:
            score: Raw prediction score (0-1)
            threshold: Classification threshold
            processing_time: Total processing time in milliseconds
            
        Returns:
            Dictionary with prediction details
        """
        # IMPORTANT: Model was trained with {'Parasitized': 0, 'Uninfected': 1}
        # So score < threshold → Parasitized, score >= threshold → Uninfected
        is_parasitized = score < threshold  # INVERTED from typical binary classification
        
        result = {
            "prediction": self._class_names[1] if is_parasitized else self._class_names[0],
            "confidence": float((1 - score) if is_parasitized else score),
            "probabilities": {
                self._class_names[0]: float(score),      # Uninfected
                self._class_names[1]: float(1 - score)   # Parasitized
            },
            "raw_score": float(score),
            "threshold": float(threshold),
            "processing_time_ms": round(processing_time, 2)
        }
        
        return result
    
    def batch_predict(
        self,
        image_paths: list,
        threshold: float = 0.5
    ) -> list:
        """
        Makes predictions for multiple images.
        
        Args:
            image_paths: List of paths to image files
            threshold: Classification threshold
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_from_path(image_path, threshold)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "prediction": None
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Returns information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            "model_loaded": self.is_model_loaded(),
            "model_path": self._model_path,
            "class_names": self._class_names
        }
        
        # Try to load version info
        version_path = 'models/model_version.json'
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                version_info = json.load(f)
                info.update(version_info)
        
        # Add model architecture info if loaded
        if self._model is not None:
            info["total_parameters"] = int(self._model.count_params())
            info["input_shape"] = self._model.input_shape[1:]
            info["output_shape"] = self._model.output_shape[1:]
        
        return info


# Global singleton instance
_prediction_service = MalariaPredictionService()


def predict_image(
    image_path: str,
    threshold: float = 0.5
) -> Dict[str, any]:
    """
    Convenience function for making predictions.
    
    Args:
        image_path: Path to the image file
        threshold: Classification threshold
        
    Returns:
        Dictionary containing prediction results
    """
    return _prediction_service.predict_from_path(image_path, threshold)


def predict_image_bytes(
    image_bytes: bytes,
    threshold: float = 0.5
) -> Dict[str, any]:
    """
    Convenience function for making predictions from bytes.
    
    Args:
        image_bytes: Raw image bytes
        threshold: Classification threshold
        
    Returns:
        Dictionary containing prediction results
    """
    return _prediction_service.predict_from_bytes(image_bytes, threshold)


def get_service() -> MalariaPredictionService:
    """
    Returns the global prediction service instance.
    
    Returns:
        MalariaPredictionService singleton
    """
    return _prediction_service


if __name__ == "__main__":
    """
    Test prediction service
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test malaria prediction')
    parser.add_argument('image_path', type=str, help='Path to test image')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    # Make prediction
    print(f"🔍 Making prediction for: {args.image_path}")
    result = predict_image(args.image_path, threshold=args.threshold)
    
    print("\n📊 Prediction Results:")
    print(json.dumps(result, indent=2))
    
    # Get model info
    service = get_service()
    info = service.get_model_info()
    print("\n📋 Model Info:")
    print(json.dumps(info, indent=2))