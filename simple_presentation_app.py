#!/usr/bin/env python3
"""
🔬 Malaria Detection Dashboard - Presentation Ready (Simplified)
Beautiful, single-page web application for malaria parasite detection
Perfect for presentations, demos, and portfolio showcase
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import os
import zipfile
from pathlib import Path
from PIL import Image
import io
from datetime import datetime

# Configure Streamlit for presentation
st.set_page_config(
    page_title="🔬 Malaria AI Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Enhanced Custom CSS for Beautiful Presentation
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main > div {
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Cards and Containers */
    .prediction-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    .status-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #10b981;
    }
    
    .insights-card {
        background: linear-gradient(135deg, #fefefe 0%, #f9fafb 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    
    /* Prediction Results */
    .result-positive {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 50%, #ef4444 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
        margin: 1rem 0;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #10b981 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        margin: 1rem 0;
    }
    
    .confidence-score {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .status-healthy {
        background: #d1fae5;
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    
    /* Processing Animation */
    .processing-animation {
        text-align: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .prediction-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Global State Management
if 'api_status' not in st.session_state:
    st.session_state.api_status = 'checking'
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'dataset_insights' not in st.session_state:
    st.session_state.dataset_insights = None

def check_api_health():
    """Check API health with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = 'healthy'
            return True
    except:
        pass
    st.session_state.api_status = 'error'
    return False

def get_model_info():
    """Get model information with caching"""
    if st.session_state.model_info is None:
        try:
            response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
            if response.status_code == 200:
                st.session_state.model_info = response.json()
        except:
            st.session_state.model_info = {}
    return st.session_state.model_info

def make_prediction(image_file):
    """Make prediction via API"""
    try:
        files = {"file": ("image.png", image_file, "image/png")}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
    return None

def analyze_image_properties_simple(image):
    """Simplified image analysis without OpenCV"""
    img_array = np.array(image)
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        # Simple RGB to grayscale conversion
        gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    else:
        gray = img_array
    
    # Calculate basic properties
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Simple texture measure (standard deviation of local means)
    h, w = gray.shape
    block_size = min(h, w) // 8
    if block_size > 0:
        texture_measure = 0
        count = 0
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                texture_measure += np.std(block)
                count += 1
        texture_complexity = texture_measure / count if count > 0 else 0
    else:
        texture_complexity = std_intensity
    
    # Edge density approximation (high-contrast pixel ratio)
    if gray.size > 1:
        diff_h = np.abs(np.diff(gray, axis=0))
        diff_v = np.abs(np.diff(gray, axis=1))
        edge_pixels = np.sum(diff_h > np.mean(diff_h) * 2) + np.sum(diff_v > np.mean(diff_v) * 2)
        edge_density = edge_pixels / gray.size
    else:
        edge_density = 0
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'texture_complexity': texture_complexity,
        'edge_density': edge_density
    }

def load_dataset_insights():
    """Load and cache dataset insights"""
    if st.session_state.dataset_insights is None:
        data_dir = Path("data")
        insights = {
            "total_images": 0,
            "parasitized_count": 0,
            "uninfected_count": 0,
            "class_distribution": {},
            "sample_analysis": {}
        }
        
        # Analyze sample images if available
        if data_dir.exists():
            for class_name in ["Parasitized", "Uninfected"]:
                class_dir = data_dir / "test" / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*.png")))
                    insights["total_images"] += count
                    insights["class_distribution"][class_name] = count
                    
                    if class_name == "Parasitized":
                        insights["parasitized_count"] = count
                    else:
                        insights["uninfected_count"] = count
                    
                    # Analyze first few images for properties
                    sample_images = list(class_dir.glob("*.png"))[:2]
                    if sample_images:
                        intensities = []
                        complexities = []
                        
                        for img_path in sample_images:
                            try:
                                img = Image.open(img_path)
                                props = analyze_image_properties_simple(img)
                                intensities.append(props['mean_intensity'])
                                complexities.append(props['texture_complexity'])
                            except:
                                continue
                        
                        if intensities:
                            insights["sample_analysis"][class_name] = {
                                "avg_intensity": np.mean(intensities),
                                "avg_complexity": np.mean(complexities)
                            }
        
        st.session_state.dataset_insights = insights
    
    return st.session_state.dataset_insights

def create_status_indicator(status, label):
    """Create a beautiful status indicator"""
    if status == 'healthy':
        return f"""
        <div class="status-indicator status-healthy">
            ✅ {label}
        </div>
        """
    else:
        return f"""
        <div class="status-indicator status-error">
            ❌ {label}
        </div>
        """

def main():
    """Main application"""
    
    # Header Section
    st.markdown('<h1 class="main-title">🔬 Malaria AI Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Medical Diagnosis • Real-time Analysis • Production Ready</p>', unsafe_allow_html=True)
    
    # Status Bar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_healthy = check_api_health()
        status_html = create_status_indicator('healthy' if api_healthy else 'error', 'API Server')
        st.markdown(status_html, unsafe_allow_html=True)
    
    with col2:
        model_info = get_model_info()
        model_status = 'healthy' if model_info and model_info.get('model_loaded') else 'error'
        status_html = create_status_indicator(model_status, 'AI Model')
        st.markdown(status_html, unsafe_allow_html=True)
    
    with col3:
        dataset_insights = load_dataset_insights()
        data_status = 'healthy' if dataset_insights['total_images'] > 0 else 'error'
        status_html = create_status_indicator(data_status, 'Dataset')
        st.markdown(status_html, unsafe_allow_html=True)
    
    with col4:
        uptime = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="status-indicator status-healthy">
            🕐 {uptime}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Layout
    left_col, right_col = st.columns([1.2, 0.8])
    
    with left_col:
        # Image Upload and Prediction Panel
        st.markdown("""
        <div class="prediction-card">
            <h2 style="margin-top: 0; color: #1e293b; font-weight: 600;">📷 Upload & Analyze</h2>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Upload a microscopic cell image for AI-powered malaria detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose cell image",
            type=["png", "jpg", "jpeg"],
            help="Upload a high-quality microscopic image of blood cells",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Image preview with analysis
            img_col1, img_col2 = st.columns([1, 1])
            
            with img_col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with img_col2:
                # Quick image analysis
                props = analyze_image_properties_simple(image)
                
                st.markdown("""
                <div class="insights-card">
                    <h4 style="margin-top: 0; color: #374151;">🔍 Image Properties</h4>
                </div>
                """, unsafe_allow_html=True)
                
                prop_cols = st.columns(2)
                with prop_cols[0]:
                    st.metric("Intensity", f"{props['mean_intensity']:.1f}")
                    st.metric("Texture", f"{props['texture_complexity']:.1f}")
                with prop_cols[1]:
                    st.metric("Sharpness", f"{props['std_intensity']:.1f}")
                    st.metric("Edge Density", f"{props['edge_density']:.3f}")
            
            # Prediction Button
            if st.button("🔬 Analyze for Malaria", type="primary", use_container_width=True):
                if not api_healthy:
                    st.error("❌ API server is not available. Please check the connection.")
                else:
                    # Processing animation
                    progress_placeholder = st.empty()
                    progress_placeholder.markdown("""
                    <div class="processing-animation">
                        <div class="spinner"></div>
                        <p style="margin-top: 1rem; color: #667eea; font-weight: 600;">🧠 AI is analyzing the image...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make prediction
                    uploaded_file.seek(0)
                    start_time = time.time()
                    result = make_prediction(uploaded_file)
                    processing_time = (time.time() - start_time) * 1000
                    
                    progress_placeholder.empty()
                    
                    if result:
                        prediction = result["prediction"]
                        confidence = result["confidence"]
                        
                        # Beautiful result display
                        if prediction == "Parasitized":
                            st.markdown(f"""
                            <div class="result-positive">
                                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                                <h2 style="margin: 0; font-size: 1.8rem;">PARASITIZED DETECTED</h2>
                                <div class="confidence-score">{confidence:.1%}</div>
                                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Malaria parasites detected in cell sample</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-negative">
                                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                                <h2 style="margin: 0; font-size: 1.8rem;">HEALTHY CELLS</h2>
                                <div class="confidence-score">{confidence:.1%}</div>
                                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">No malaria parasites detected</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed metrics
                        st.markdown("### 📊 Analysis Details")
                        metric_cols = st.columns(4)
                        
                        with metric_cols[0]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{confidence:.1%}</div>
                                <div class="metric-label">Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[1]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{processing_time:.0f}ms</div>
                                <div class="metric-label">Speed</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[2]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{result['raw_score']:.3f}</div>
                                <div class="metric-label">Raw Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[3]:
                            risk_level = "HIGH" if prediction == "Parasitized" else "LOW"
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{risk_level}</div>
                                <div class="metric-label">Risk Level</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Probability visualization
                        prob_df = pd.DataFrame({
                            "Class": list(result["probabilities"].keys()),
                            "Probability": list(result["probabilities"].values())
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x="Class",
                            y="Probability",
                            color="Class",
                            color_discrete_map={"Uninfected": "#10b981", "Parasitized": "#ef4444"},
                            title="🎯 Class Probabilities"
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        # Data Insights Section
        st.markdown("""
        <div class="insights-card">
            <h2 style="margin-top: 0; color: #1e293b; font-weight: 600;">📊 Dataset Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        
        insights = load_dataset_insights()
        
        if insights['total_images'] > 0:
            # Dataset Overview
            total_col1, total_col2 = st.columns(2)
            
            with total_col1:
                st.metric("📊 Total Images", insights['total_images'])
            with total_col2:
                balance_ratio = insights['parasitized_count'] / max(insights['uninfected_count'], 1)
                st.metric("⚖️ Balance Ratio", f"{balance_ratio:.2f}")
            
            # Class distribution pie chart
            if insights['class_distribution']:
                fig_pie = px.pie(
                    values=list(insights['class_distribution'].values()),
                    names=list(insights['class_distribution'].keys()),
                    title="🥧 Class Distribution",
                    color_discrete_map={"Uninfected": "#10b981", "Parasitized": "#ef4444"}
                )
                fig_pie.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Sample analysis comparison
            if insights['sample_analysis']:
                st.markdown("### 🔬 Sample Analysis")
                
                analysis_data = []
                for class_name, props in insights['sample_analysis'].items():
                    analysis_data.extend([
                        {"Class": class_name, "Metric": "Avg Intensity", "Value": props['avg_intensity']},
                        {"Class": class_name, "Metric": "Texture Complexity", "Value": props['avg_complexity']}
                    ])
                
                if analysis_data:
                    analysis_df = pd.DataFrame(analysis_data)
                    
                    fig_analysis = px.bar(
                        analysis_df,
                        x="Metric",
                        y="Value",
                        color="Class",
                        barmode="group",
                        color_discrete_map={"Uninfected": "#10b981", "Parasitized": "#ef4444"},
                        title="📈 Image Properties Comparison"
                    )
                    fig_analysis.update_layout(height=300)
                    st.plotly_chart(fig_analysis, use_container_width=True)
        else:
            st.info("🔄 No dataset found. Upload some sample images to see insights!")
        
        # Model Information
        st.markdown("""
        <div class="insights-card">
            <h2 style="margin-top: 0; color: #1e293b; font-weight: 600;">🧠 Model Information</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if model_info:
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.metric("🎯 Architecture", "MobileNetV2")
                if 'total_parameters' in model_info:
                    params = model_info['total_parameters']
                    st.metric("⚙️ Parameters", f"{params:,}")
            
            with info_col2:
                if 'final_val_accuracy' in model_info:
                    acc = model_info['final_val_accuracy']
                    st.metric("📈 Accuracy", f"{acc:.1%}")
                st.metric("🚀 Status", "Ready")
        else:
            st.warning("🔄 Connect to API server to see model information")
        
        # Quick Demo Samples
        st.markdown("""
        <div class="insights-card">
            <h2 style="margin-top: 0; color: #1e293b; font-weight: 600;">🧪 Quick Demo</h2>
            <p style="color: #64748b;">Try these sample images for instant testing:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load sample images
        sample_images = []
        data_dir = Path("data/test")
        if data_dir.exists():
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    for img_file in list(class_dir.glob("*.png"))[:2]:
                        sample_images.append({
                            "name": f"{class_dir.name}: {img_file.name}",
                            "path": str(img_file),
                            "class": class_dir.name
                        })
        
        if sample_images:
            for i, sample in enumerate(sample_images[:4]):
                if st.button(f"🔬 Test: {sample['class']}", key=f"sample_{i}", use_container_width=True):
                    if api_healthy:
                        try:
                            with open(sample["path"], "rb") as f:
                                result = make_prediction(f)
                                if result:
                                    pred_class = result['prediction']
                                    confidence = result['confidence']
                                    is_correct = pred_class == sample['class']
                                    
                                    status = "✅ Correct" if is_correct else "❌ Incorrect"
                                    st.success(f"{status}: {pred_class} ({confidence:.1%})")
                        except Exception as e:
                            st.error(f"Error testing sample: {str(e)}")
                    else:
                        st.warning("🔄 Please start the API server to test samples")
        else:
            st.info("📁 No sample images found. Add some test images to the data/test/ folder.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem;">
        🔬 Built with ❤️ for medical AI • <strong>Malaria Detection System</strong> • 
        <a href="https://github.com/MichaelMusembi/Malaria-Detection-Pipeline" style="color: #667eea;">View Source Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
