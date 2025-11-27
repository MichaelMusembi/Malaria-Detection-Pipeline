"""
Test Suite for Malaria Detection Prediction Service

Tests the prediction functionality with real malaria images.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to Python path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import MalariaPredictionService
from preprocessing import preprocess_single_image


class TestMalariaPrediction:
    """Test suite for malaria prediction functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with prediction service."""
        cls.prediction_service = MalariaPredictionService()
        cls.test_data_dir = Path(__file__).parent.parent / "data" / "test"
    
    def test_service_initialization(self):
        """Test that prediction service initializes properly."""
        assert self.prediction_service is not None
        assert hasattr(self.prediction_service, 'predict_image')
    
    def test_model_loading(self):
        """Test that the model loads successfully."""
        try:
            # This should trigger model loading
            result = self.prediction_service.get_model_info()
            assert result is not None
        except FileNotFoundError:
            pytest.skip("Model file not found - run training first")
    
    def test_parasitized_prediction(self):
        """Test prediction on parasitized samples."""
        parasitized_dir = self.test_data_dir / "Parasitized"
        if not parasitized_dir.exists():
            pytest.skip("Test data not found")
        
        sample_files = list(parasitized_dir.glob("*.png"))[:2]  # Test first 2 files
        
        for sample_file in sample_files:
            result = self.prediction_service.predict_image(str(sample_file))
            
            assert result is not None
            assert 'prediction' in result
            assert 'confidence' in result
            assert 'processing_time' in result
            assert result['confidence'] >= 0.0
            assert result['confidence'] <= 1.0
    
    def test_uninfected_prediction(self):
        """Test prediction on uninfected samples."""
        uninfected_dir = self.test_data_dir / "Uninfected"
        if not uninfected_dir.exists():
            pytest.skip("Test data not found")
        
        sample_files = list(uninfected_dir.glob("*.png"))[:2]  # Test first 2 files
        
        for sample_file in sample_files:
            result = self.prediction_service.predict_image(str(sample_file))
            
            assert result is not None
            assert 'prediction' in result
            assert 'confidence' in result
            assert 'processing_time' in result
            assert result['confidence'] >= 0.0
            assert result['confidence'] <= 1.0
    
    def test_preprocessing(self):
        """Test image preprocessing functionality."""
        test_file = self.test_data_dir / "simple_test.png"
        if test_file.exists():
            processed = preprocess_single_image(str(test_file))
            
            assert processed is not None
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float32
            assert np.min(processed) >= 0.0
            assert np.max(processed) <= 1.0
    
    def test_invalid_image_handling(self):
        """Test handling of invalid image files."""
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, ValueError)):
            self.prediction_service.predict_image("non_existent_file.jpg")


if __name__ == "__main__":
    pytest.main([__file__])
