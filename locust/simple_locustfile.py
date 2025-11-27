"""
Simple Locust Load Test for Malaria Detection API
"""

from locust import HttpUser, task, between
import os
from pathlib import Path


class MalariaAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Load test images on user start."""
        self.test_images = []
        
        # Get test images
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data" / "test"
        
        # Load all PNG files
        for img_file in data_dir.rglob("*.png"):
            if img_file.is_file():
                self.test_images.append(str(img_file))
        
        print(f"Loaded {len(self.test_images)} test images")
    
    @task(3)
    def health_check(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
    
    @task(10)
    def predict_image(self):
        """Test image prediction."""
        if not self.test_images:
            return
        
        # Pick random image
        import random
        image_path = random.choice(self.test_images)
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/png')}
                response = self.client.post("/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'prediction' not in data:
                        print(f"Invalid response structure: {data}")
                else:
                    print(f"Prediction failed: {response.status_code}")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Run with: locust -f simple_locustfile.py --host=http://localhost:8090")
