"""
Test Suite for Malaria Detection API Endpoints

Tests the FastAPI application endpoints with real images.
"""

import os
import sys
import pytest
import requests
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080"


class TestMalariaAPI:
    """Test suite for malaria detection API."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.base_url = API_BASE_URL
        cls.test_data_dir = Path(__file__).parent.parent / "data" / "test"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
        except requests.ConnectionError:
            pytest.skip("API server not running on localhost:8080")
    
    def test_predict_endpoint_parasitized(self):
        """Test prediction endpoint with parasitized image."""
        try:
            parasitized_dir = self.test_data_dir / "Parasitized"
            if not parasitized_dir.exists():
                pytest.skip("Test data not found")
            
            sample_file = next(parasitized_dir.glob("*.png"))
            
            with open(sample_file, 'rb') as f:
                files = {'file': ('test.png', f, 'image/png')}
                response = requests.post(f"{self.base_url}/predict", files=files)
            
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "processing_time" in data
            
        except requests.ConnectionError:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_uninfected(self):
        """Test prediction endpoint with uninfected image."""
        try:
            uninfected_dir = self.test_data_dir / "Uninfected"
            if not uninfected_dir.exists():
                pytest.skip("Test data not found")
            
            sample_file = next(uninfected_dir.glob("*.png"))
            
            with open(sample_file, 'rb') as f:
                files = {'file': ('test.png', f, 'image/png')}
                response = requests.post(f"{self.base_url}/predict", files=files)
            
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "processing_time" in data
            
        except requests.ConnectionError:
            pytest.skip("API server not running")
    
    def test_predict_invalid_file(self):
        """Test prediction endpoint with invalid file."""
        try:
            # Test with empty file
            files = {'file': ('test.txt', b'invalid data', 'text/plain')}
            response = requests.post(f"{self.base_url}/predict", files=files)
            
            assert response.status_code in [400, 422]  # Bad request or validation error
            
        except requests.ConnectionError:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__])
