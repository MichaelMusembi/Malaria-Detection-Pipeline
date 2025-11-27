"""
Streamlit UI Dashboard for Malaria Detection Pipeline

Provides a user-friendly interface for predictions and model management.
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Configuration
API_URL = "http://localhost:8000"


def check_api_health():
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None


def get_uptime():
    """Get API uptime information"""
    try:
        response = requests.get(f"{API_URL}/uptime", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None


def make_prediction(image_file, threshold=0.5):
    """Make prediction via API"""
    try:
        files = {"file": image_file}
        params = {"threshold": threshold}
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)


def trigger_retraining(zip_file, epochs=10, batch_size=32):
    """Trigger model retraining via API"""
    try:
        files = {"file": zip_file}
        params = {"epochs": epochs, "batch_size": batch_size}
        response = requests.post(
            f"{API_URL}/retrain",
            files=files,
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None


# Page configuration
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .parasitized {
        background-color: #ffebee;
        color: #c62828;
    }
    .uninfected {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


# Main title
st.markdown('<div class="main-header">🦠 Malaria Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Blood Cell Analysis</div>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🔍 Prediction", "🔄 Retrain Model", "📊 Dashboard", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    # API Status
    st.subheader("🔌 API Status")
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ Connected")
        if health_data:
            st.metric("Model Version", health_data.get("model_version", "N/A"))
            uptime = health_data.get("uptime_seconds", 0)
            st.metric("Uptime", f"{uptime/3600:.1f}h")
    else:
        st.error("❌ Disconnected")
        st.info("Make sure the API is running:\n```bash\npython src/api.py\n```")


# Page: Prediction
if page == "🔍 Prediction":
    st.header("🔍 Make a Prediction")
    st.write("Upload a blood cell image to detect malaria parasites")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a blood cell microscopy image"
        )
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    success, result = make_prediction(uploaded_file, threshold)
                    
                    if success:
                        st.session_state['last_prediction'] = result
                    else:
                        st.error(f"❌ Prediction failed: {result}")
    
    with col2:
        st.subheader("Results")
        
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            
            # Prediction result
            prediction = result['prediction']
            confidence = result['confidence']
            
            result_class = "parasitized" if prediction == "Parasitized" else "uninfected"
            st.markdown(
                f'<div class="prediction-result {result_class}">'
                f'{prediction}<br>'
                f'<span style="font-size: 1.5rem;">{confidence*100:.1f}% Confidence</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Probabilities
            st.subheader("Class Probabilities")
            probs = result['probabilities']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=['#2e7d32', '#c62828'],
                    text=[f'{v*100:.1f}%' for v in probs.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("Performance Metrics")
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Processing Time", f"{result['processing_time_ms']:.1f} ms")
            
            with metric_col2:
                st.metric("Raw Score", f"{result.get('raw_score', 0):.4f}")
            
            # JSON output
            with st.expander("📄 View Full JSON Response"):
                st.json(result)
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")


# Page: Retrain Model
elif page == "🔄 Retrain Model":
    st.header("🔄 Retrain Model")
    st.write("Upload new training data to update the model")
    
    st.warning("⚠️ Retraining will temporarily increase server load. The process runs in the background.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=50,
            value=10,
            help="More epochs = better accuracy but longer training"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64],
            index=2,
            help="Higher batch size = faster training but more memory"
        )
        
        st.markdown("---")
        
        st.subheader("Upload Training Data")
        st.info("📦 ZIP file must contain:\n- `Parasitized/` folder\n- `Uninfected/` folder")
        
        zip_file = st.file_uploader(
            "Upload ZIP file",
            type=['zip'],
            help="Training data in required folder structure"
        )
        
        if zip_file is not None:
            st.success(f"✅ File loaded: {zip_file.name} ({zip_file.size / 1024 / 1024:.1f} MB)")
            
            if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
                with st.spinner("Starting retraining..."):
                    zip_file.seek(0)
                    success, result = trigger_retraining(zip_file, epochs, batch_size)
                    
                    if success:
                        st.success(f"✅ Retraining started!")
                        st.info(f"Task ID: {result['task_id']}")
                        st.session_state['retrain_task_id'] = result['task_id']
                    else:
                        st.error(f"❌ Failed to start retraining: {result}")
    
    with col2:
        st.subheader("Retraining Status")
        
        if 'retrain_task_id' in st.session_state:
            task_id = st.session_state['retrain_task_id']
            
            if st.button("🔄 Refresh Status"):
                st.rerun()
            
            try:
                response = requests.get(f"{API_URL}/retrain/status/{task_id}")
                if response.status_code == 200:
                    status = response.json()
                    
                    status_emoji = {
                        "started": "🟡",
                        "running": "🔵",
                        "completed": "🟢",
                        "failed": "🔴"
                    }
                    
                    st.markdown(f"### {status_emoji.get(status['status'], '⚪')} Status: {status['status'].upper()}")
                    
                    if status['status'] == 'completed':
                        st.success("✅ Retraining completed successfully!")
                        
                        if 'metrics' in status:
                            metrics = status['metrics']
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric("Accuracy", f"{metrics['val_accuracy']*100:.2f}%")
                            
                            with metric_col2:
                                st.metric("Precision", f"{metrics['val_precision']*100:.2f}%")
                            
                            with metric_col3:
                                st.metric("Recall", f"{metrics['val_recall']*100:.2f}%")
                        
                        st.info("🔄 Reload the model in the Dashboard to use the new version")
                    
                    elif status['status'] == 'failed':
                        st.error(f"❌ Retraining failed: {status.get('error', 'Unknown error')}")
                    
                    with st.expander("📄 View Full Status"):
                        st.json(status)
                else:
                    st.warning("Task not found")
            except:
                st.error("Failed to fetch status")
        else:
            st.info("No active retraining task")


# Page: Dashboard
elif page == "📊 Dashboard":
    st.header("📊 System Dashboard")
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uptime_data = get_uptime()
        if uptime_data:
            st.metric(
                "⏱️ System Uptime",
                uptime_data['uptime_formatted']
            )
        else:
            st.metric("⏱️ System Uptime", "N/A")
    
    with col2:
        model_info = get_model_info()
        if model_info:
            st.metric(
                "🤖 Model Version",
                model_info.get('version', 'N/A')
            )
        else:
            st.metric("🤖 Model Version", "N/A")
    
    with col3:
        is_healthy, health_data = check_api_health()
        status_text = "🟢 Healthy" if is_healthy else "🔴 Down"
        st.metric("💚 API Status", status_text)
    
    st.markdown("---")
    
    # Model information
    if model_info:
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            info_data = {
                "Trained Date": model_info.get('trained_date', 'N/A'),
                "Total Parameters": f"{model_info.get('total_parameters', 0):,}",
                "Input Shape": str(model_info.get('input_shape', 'N/A')),
                "Output Shape": str(model_info.get('output_shape', 'N/A'))
            }
            
            for key, value in info_data.items():
                st.text(f"{key}: {value}")
        
        with col2:
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': list(metrics.values())
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Reload model button
        if st.button("🔄 Reload Model", type="secondary"):
            with st.spinner("Reloading model..."):
                try:
                    response = requests.post(f"{API_URL}/model/reload")
                    if response.status_code == 200:
                        st.success("✅ Model reloaded successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to reload model")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# Page: About
elif page == "ℹ️ About":
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🦠 Malaria Detection System
    
    This AI-powered system automatically detects malaria parasites in blood cell microscopy images
    using deep learning and transfer learning techniques.
    
    #### 🎯 Features
    - **Real-time Predictions**: Analyze blood cell images in seconds
    - **High Accuracy**: Powered by MobileNetV2 architecture
    - **Continuous Learning**: Retrain with new data to improve performance
    - **RESTful API**: Easy integration with other systems
    - **Interactive Dashboard**: User-friendly interface for medical professionals
    
    #### 🏗️ Technology Stack
    - **Deep Learning**: TensorFlow 2.15, Keras
    - **Architecture**: MobileNetV2 (Transfer Learning)
    - **API**: FastAPI, Uvicorn
    - **UI**: Streamlit
    - **Containerization**: Docker, Docker Compose
    
    #### 📊 Dataset
    - **Source**: NIH Kaggle Dataset - Cell Images for Detecting Malaria
    - **Classes**: Parasitized vs Uninfected
    - **Total Images**: ~27,558 microscopy images
    
    #### 🎓 Model Performance
    The model achieves high accuracy in distinguishing between:
    - **Parasitized cells**: Infected with malaria parasites
    - **Uninfected cells**: Healthy blood cells
    
    #### 🔒 Privacy & Security
    - All images are processed locally
    - No data is stored permanently
    - HIPAA-compliant deployment available
    
    #### 📝 Usage Instructions
    1. **Prediction**: Upload a blood cell image to get instant results
    2. **Retraining**: Upload new labeled data to improve the model
    3. **Monitoring**: Track system health and performance metrics
    
    #### 👥 Developed By
    Machine Learning Engineering Team
    
    #### 📄 License
    MIT License - Open Source
    
    ---
    
    **⚠️ Medical Disclaimer**: This system is for research and educational purposes.
    Always consult qualified medical professionals for diagnosis and treatment.
    """)
    
    # Sample predictions showcase
    st.subheader("📸 Sample Results")
    st.info("Upload your own images in the 'Prediction' tab to see live results!")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Malaria Detection System v1.0.0 | Built with ❤️ using TensorFlow & FastAPI"
    "</div>",
    unsafe_allow_html=True
)
