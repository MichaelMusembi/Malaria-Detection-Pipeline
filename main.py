#!/usr/bin/env python3
"""
Vercel-optimized API for Malaria Detection Pipeline
This version uses model compression for serverless deployment
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# For Vercel, we'll use a lightweight model approach
os.environ["USE_LIGHTWEIGHT_MODEL"] = "true"

# Import the FastAPI app from src
try:
    from api import app
except ImportError:
    # Fallback for Vercel deployment
    import api as api_module
    app = api_module.app

# Export the app for Vercel
app = app

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
