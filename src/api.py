"""
FastAPI Application for Malaria Detection Pipeline

Provides REST API endpoints for predictions, retraining, and monitoring.
"""

import os
import time
import tempfile
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prediction import get_service as get_prediction_service
from retrain_worker import get_worker, start_retraining


# Track server start time for uptime calculation
SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup: Load model
    print("🚀 Starting Malaria Detection API...")
    service = get_prediction_service()
    try:
        # Use MODEL_PATH environment variable if set
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            # Default path for Docker container (running from /app/src)
            model_path = "../models/malaria_mobilenet.keras"
        service.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model: {e}")
        print("   Model will be loaded on first prediction request")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Malaria Detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Malaria Parasite Detection API",
    description="REST API for automated malaria detection in blood cell images",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    raw_score: Optional[float] = None


class RetrainResponse(BaseModel):
    message: str
    task_id: str
    status: str


class UptimeResponse(BaseModel):
    uptime_seconds: float
    uptime_formatted: str
    started_at: str


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Malaria Parasite Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "uptime": "/uptime",
            "predict": "/predict",
            "retrain": "/retrain",
            "docs": "/docs"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    service = get_prediction_service()
    model_info = service.get_model_info()
    
    uptime = time.time() - SERVER_START_TIME
    
    return HealthResponse(
        status="healthy",
        model_loaded=service.is_model_loaded(),
        model_version=model_info.get("version"),
        uptime_seconds=uptime
    )


# Uptime endpoint
@app.get("/uptime", response_model=UptimeResponse)
async def get_uptime():
    """
    Returns server uptime information.
    
    Returns:
        Uptime in seconds and formatted string
    """
    uptime_seconds = time.time() - SERVER_START_TIME
    
    # Format uptime
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    uptime_formatted = f"{hours}h {minutes}m {seconds}s"
    
    started_at = datetime.fromtimestamp(SERVER_START_TIME).isoformat()
    
    return UptimeResponse(
        uptime_seconds=uptime_seconds,
        uptime_formatted=uptime_formatted,
        started_at=started_at
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5
):
    """
    Makes a malaria detection prediction on an uploaded image.
    
    Args:
        file: Uploaded image file (PNG, JPEG)
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Prediction results with confidence scores
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPEG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        service = get_prediction_service()
        result = service.predict_from_bytes(image_bytes, threshold=threshold)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...)
):
    """
    Makes predictions on multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images allowed per batch request"
        )
    
    results = []
    service = get_prediction_service()
    
    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image",
                    "prediction": None
                })
                continue
            
            image_bytes = await file.read()
            result = service.predict_from_bytes(image_bytes)
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "prediction": None
            })
    
    return {"results": results, "total": len(files)}


# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """
    Returns information about the loaded model.
    
    Returns:
        Model metadata and version information
    """
    service = get_prediction_service()
    return service.get_model_info()


# Model reload endpoint
@app.post("/model/reload")
async def reload_model():
    """
    Forces a model reload (useful after retraining).
    
    Returns:
        Success message
    """
    try:
        service = get_prediction_service()
        service.reload_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


# Retraining endpoint
@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Triggers model retraining with new data.
    
    Expects a ZIP file with the following structure:
    - Parasitized/ (folder with infected cell images)
    - Uninfected/ (folder with uninfected cell images)
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded ZIP file with training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Task information
    """
    # Validate file type
    if not file.content_type in ['application/zip', 'application/x-zip-compressed']:
        raise HTTPException(
            status_code=400,
            detail="File must be a ZIP archive"
        )
    
    try:
        # Generate task ID
        task_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save uploaded ZIP to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            zip_path = tmp_file.name
        
        # Start retraining in background
        background_tasks.add_task(
            start_retraining,
            zip_path=zip_path,
            task_id=task_id,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Also cleanup ZIP file after task completes
        background_tasks.add_task(os.remove, zip_path)
        
        return RetrainResponse(
            message="Retraining started in background",
            task_id=task_id,
            status="started"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start retraining: {str(e)}"
        )


# Retraining status endpoint
@app.get("/retrain/status/{task_id}")
async def get_retrain_status(task_id: str):
    """
    Gets the status of a retraining task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        Task status information
    """
    worker = get_worker()
    status = worker.get_retrain_status(task_id)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return status


# List all retraining tasks
@app.get("/retrain/tasks")
async def list_retrain_tasks():
    """
    Lists all retraining tasks.
    
    Returns:
        Dictionary of all task statuses
    """
    worker = get_worker()
    return worker.list_retrain_tasks()


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )