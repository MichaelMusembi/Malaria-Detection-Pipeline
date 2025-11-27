# 🚀 Render Deployment Guide
### Deploy Malaria Detection System with One Click!

## 📋 Pre-Deployment Checklist
- ✅ GitHub repository with render.yaml
- ✅ Docker containers configured 
- ✅ API and UI services ready
- ✅ Git LFS for large model files

## 🎯 Quick Deploy (Recommended)

### Option 1: Blueprint Deploy (One Click)
1. **Visit Render Dashboard**: [render.com/dashboard](https://dashboard.render.com)
2. **Create New Blueprint**: 
   - Click "New +" → "Blueprint"
   - Repository: `MichaelMusembi/Malaria-Detection-Pipeline`
   - Branch: `main` 
   - Blueprint Path: `render.yaml`
3. **Deploy**: Click "Apply" - Both services will deploy automatically!

### Option 2: Manual Deploy

#### 🔧 Deploy API Service First
1. **New Web Service**:
   - Repository: `MichaelMusembi/Malaria-Detection-Pipeline`
   - Name: `malaria-detection-api`
   - Runtime: `Docker`
   - Dockerfile Path: `./docker/Dockerfile`
   - Region: `Oregon (US West)`

2. **Environment Variables**:
   ```
   ENVIRONMENT=production
   TF_CPP_MIN_LOG_LEVEL=2
   PYTHONUNBUFFERED=1
   MODEL_PATH=/app/models/malaria_mobilenet.keras
   ```

3. **Advanced Settings**:
   - Health Check Path: `/health`
   - Build Command: (leave empty for Docker)
   - Start Command: (leave empty for Docker)

#### 🎨 Deploy UI Service Second  
1. **New Web Service**:
   - Repository: `MichaelMusembi/Malaria-Detection-Pipeline` 
   - Name: `malaria-detection-ui`
   - Runtime: `Docker`
   - Dockerfile Path: `./docker/Dockerfile.ui`
   - Region: `Oregon (US West)`

2. **Environment Variables**:
   ```
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_ADDRESS=0.0.0.0  
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   API_BASE_URL=https://malaria-detection-api.onrender.com
   ```

3. **Advanced Settings**:
   - Health Check Path: `/_stcore/health`
   - Build Command: (leave empty for Docker)
   - Start Command: (leave empty for Docker)

## 🔗 Expected URLs
After deployment, you'll get:
- **API**: `https://malaria-detection-api.onrender.com`
- **UI**: `https://malaria-detection-ui.onrender.com`

## ⚡ Quick Testing
Once deployed, test these endpoints:
- **API Health**: `https://malaria-detection-api.onrender.com/health`
- **API Docs**: `https://malaria-detection-api.onrender.com/docs`  
- **UI App**: `https://malaria-detection-ui.onrender.com`

## 🐛 Troubleshooting

### Common Issues:
1. **Build Timeout**: 
   - Git LFS files may take time to download
   - Check build logs for TensorFlow installation
   
2. **Health Check Fails**:
   - API: Check model loading in logs
   - UI: Verify Streamlit starts properly
   
3. **UI Can't Connect to API**:
   - Verify API_BASE_URL in UI environment
   - Check API service is fully deployed first

### Deployment Logs:
- Monitor "Events" tab for each service
- Check "Logs" for runtime errors
- Look for "Health check passed" messages

## 🎊 Success!
When both services show "Live" status:
1. **API Test**: Visit `/health` endpoint
2. **UI Test**: Upload a cell image for prediction  
3. **Integration**: Verify UI calls API successfully
4. **Performance**: Run load test with Locust (optional)

## 📱 Share Your Links
Once deployed, share these public URLs:
- **Demo the UI**: `https://malaria-detection-ui.onrender.com`
- **API Documentation**: `https://malaria-detection-api.onrender.com/docs`

Perfect for presentations, portfolios, and academic submissions! 🎓

---
💡 **Tip**: Bookmark your service URLs and enable "Auto-Deploy" for automatic updates when you push to GitHub.
