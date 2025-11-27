# 🦠 Malaria Detection Pipeline

## Overview
An end-to-end machine learning pipeline for detecting malaria parasites in blood cell microscopy images using deep learning and transfer learning techniques.

## 🎯 Project Description
This AI-powered system automatically detects malaria parasites in blood cell images using a MobileNetV2-based transfer learning model. The system provides real-time predictions, model retraining capabilities, and comprehensive monitoring through a web interface.

## 🏗️ Architecture
- **Deep Learning Model**: MobileNetV2 (Transfer Learning)
- **Framework**: TensorFlow/Keras
- **API**: FastAPI with auto-generated documentation
- **Frontend**: Streamlit dashboard
- **Data**: NIH Kaggle Dataset - Cell Images for Detecting Malaria
- **Deployment**: Docker + Docker Compose + Nginx

## 📊 Dataset
- **Source**: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Classes**: Parasitized vs Uninfected
- **Total Images**: ~27,558 microscopy images
- **Resolution**: Various sizes, normalized to 224x224

## 🚀 Features

### 🔮 ML Pipeline
- **Data Acquisition**: Automated Kaggle dataset download
- **Data Processing**: Image preprocessing and augmentation
- **Model Training**: MobileNetV2 transfer learning
- **Model Testing**: Comprehensive evaluation metrics
- **Model Retraining**: Background retraining with new data

### 🌐 API Endpoints
- `POST /predict` - Single image prediction
- `POST /retrain` - Trigger model retraining
- `GET /health` - System health check
- `GET /model/info` - Model metadata
- `POST /model/reload` - Reload trained model

### 🖥️ User Interface
- **Real-time Predictions**: Upload and analyze images instantly
- **Model Monitoring**: Live system metrics and uptime
- **Data Visualization**: Interactive charts and graphs
- **Bulk Upload**: Retrain with new datasets
- **Performance Metrics**: Processing time and accuracy stats

### ⚖️ Load Balancing & Scaling
- **Nginx**: Load balancer for multiple API instances
- **Docker Compose**: Multi-container orchestration
- **Horizontal Scaling**: Add more API containers as needed

## 📁 Project Structure

```
Malaria-Detection-Pipeline/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
│
├── notebook/
│   └── malaria-detection-pipeline.ipynb  # Analysis & training notebook
│
├── src/
│   ├── preprocessing.py         # Image preprocessing utilities
│   ├── model.py                # Model training and evaluation
│   ├── prediction.py           # Prediction service
│   ├── api.py                  # FastAPI application
│   ├── ui.py                   # Streamlit dashboard
│   └── retrain_worker.py       # Background retraining
│
├── data/
│   ├── train/                  # Training data
│   └── test/                   # Test data
│
├── models/
│   └── malaria_mobilenet.h5    # Trained model
│
├── docker/
│   ├── Dockerfile              # Container configuration
│   ├── docker-compose.yml      # Multi-service setup
│   └── nginx.conf              # Load balancer config
│
├── locust/
│   └── locustfile.py           # Load testing configuration
│
└── tests/
    └── test_prediction.py      # Unit tests
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized deployment)

### 1. Clone Repository
```bash
git clone https://github.com/MichaelMusembi/Malaria-Detection-Pipeline.git
cd Malaria-Detection-Pipeline
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
# Organize data into data/train/ and data/test/ folders
```

### 4. Train Model (Optional)
```bash
# If you want to retrain the model
python src/model.py --train-dir data/train --epochs 20
```

## 🚀 Running the Application

### Option 1: Local Development
```bash
# Terminal 1 - Start API
python src/api.py

# Terminal 2 - Start UI  
streamlit run src/ui.py

# Access the application
# API Documentation: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Load Balanced API: http://localhost:80
# Direct API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### Option 3: Production Scaling
```bash
# Scale API instances
docker-compose up --scale api=3

# With load balancer
docker-compose -f docker-compose.yml up nginx api
```

## 📊 Model Performance

### Training Metrics
- **Accuracy**: ~96%
- **Precision**: ~95%
- **Recall**: ~97%
- **F1-Score**: ~96%
- **AUC**: ~0.98

### Inference Performance
- **Single Prediction**: ~200ms
- **Batch Processing**: ~50ms/image
- **Model Loading**: ~3-5 seconds

## 🧪 Load Testing

### Run Load Tests
```bash
# Start API first
python src/api.py

# Run Locust tests
locust -f locust/locustfile.py --host=http://localhost:8000

# Access Locust UI: http://localhost:8089
```

### Performance Benchmarks
- **1 Container**: 50 RPS, 400ms avg response
- **3 Containers**: 120 RPS, 250ms avg response  
- **5 Containers**: 180 RPS, 200ms avg response

## 📈 Usage Examples

### API Usage
```python
import requests

# Single prediction
with open('cell_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'threshold': 0.5}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

### Python SDK Usage
```python
from src.prediction import predict_image

# Local prediction
result = predict_image('path/to/image.jpg', threshold=0.5)
print(result)
```

## 🔒 Security & Privacy
- No permanent image storage
- Local processing (HIPAA compliant)
- API rate limiting
- Input validation and sanitization

## 📺 Demo Video
[🎥 Watch Demo on YouTube](https://youtube.com/placeholder-link)

## 🤝 Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- **Michael Musembi** - Initial work

## 🙏 Acknowledgments
- NIH for the malaria cell image dataset
- TensorFlow team for the framework
- Kaggle for dataset hosting
- African Leadership University for the project guidance

## ⚠️ Medical Disclaimer
This system is for research and educational purposes only. Always consult qualified medical professionals for diagnosis and treatment.

---

**Built with ❤️ using TensorFlow, FastAPI, and Streamlit**