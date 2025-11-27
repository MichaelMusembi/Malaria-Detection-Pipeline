"""
Simple Locust Load Testing for Malaria Detection API
"""

import random
import os
from locust import HttpUser, task, between


class MalariaAPIUser(HttpUser):
    """Simulate users interacting with the malaria detection API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        print("🚀 Starting load test user...")
        
        # Test that the API is healthy before starting load test
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"❌ API health check failed: {response.status_code}")
    
    @task(5)
    def health_check(self):
        """Test the health endpoint (frequent operation)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"✅ Health check successful")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(3)
    def get_model_info(self):
        """Test the model info endpoint."""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"✅ Model info retrieved")
            else:
                response.failure(f"Model info failed: {response.status_code}")
    
    @task(1)
    def get_docs(self):
        """Test the API documentation endpoint."""
        with self.client.get("/docs", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Docs failed: {response.status_code}")


if __name__ == "__main__":
    print("🚀 Simple Malaria Detection API Load Testing")
    print("=" * 50)
    print("Run with: locust -f locustfile_simple.py --host=http://localhost:8000")
