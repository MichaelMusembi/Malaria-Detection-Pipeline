# 🚀 Vercel Deployment Guide - Malaria Detection Pipeline

## Quick Deploy to Vercel

### Prerequisites
- ✅ Vercel account (free at vercel.com)
- ✅ GitHub repository pushed
- ✅ All files committed

### Option 1: One-Click Deploy

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect `vercel.json` configuration
   - Click "Deploy"

### Option 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   vercel --prod
   ```

## 🔧 Configuration Files

### vercel.json
- ✅ **Main API**: Configured for FastAPI deployment
- ✅ **Routes**: All API endpoints mapped
- ✅ **Timeout**: 30 seconds for ML inference
- ✅ **Python Path**: Configured for src/ directory

### main.py
- ✅ **Entry Point**: Vercel-optimized wrapper
- ✅ **Path Management**: Handles src/ imports
- ✅ **Fallback**: Local testing capability

### requirements-vercel.txt
- ✅ **Dependencies**: Optimized for serverless
- ✅ **TensorFlow**: Latest compatible version
- ✅ **FastAPI**: Production-ready setup

## 🌐 Expected Deployment URLs

After deployment, your API will be available at:
- **API Base**: `https://your-project-name.vercel.app`
- **Health Check**: `https://your-project-name.vercel.app/health`
- **Predictions**: `https://your-project-name.vercel.app/predict`
- **API Docs**: `https://your-project-name.vercel.app/docs`

## 📱 Frontend Deployment

For the Streamlit UI, you have options:

### Option 1: Streamlit Cloud
1. **Connect GitHub**: Link your repository
2. **Set Main File**: `simple_presentation_app.py`
3. **Environment Variables**: Add `API_BASE_URL=https://your-vercel-api.vercel.app`

### Option 2: Separate Vercel Deployment
Create a separate `vercel-frontend.json`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "simple_presentation_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "simple_presentation_app.py"
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Files Too Large**
   - **Solution**: Use Git LFS for `.h5` and `.keras` files
   - **Command**: `git lfs track "*.h5" "*.keras"`

2. **Import Errors**
   - **Solution**: Check `PYTHONPATH` in `vercel.json`
   - **Verify**: All dependencies in `requirements-vercel.txt`

3. **Timeout Issues**
   - **Solution**: Increase `maxDuration` in `vercel.json`
   - **Alternative**: Optimize model loading

### Testing Deployment

```bash
# Test local main.py
python main.py

# Test API endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "file=@test_image.png"
```

## 📊 Performance Considerations

### Serverless Limitations
- **Cold Start**: First request may be slow (~10s)
- **Memory**: Limited to 1GB on free tier
- **Timeout**: 30 seconds maximum
- **Storage**: Ephemeral (models reload each time)

### Optimizations
- ✅ **Lightweight Models**: MobileNetV2 preferred
- ✅ **Model Caching**: Implemented in prediction service
- ✅ **Fast Dependencies**: opencv-python-headless instead of opencv-python
- ✅ **Efficient Imports**: Lazy loading where possible

## 🎯 Next Steps After Deployment

1. **Test All Endpoints**
   ```bash
   curl https://your-app.vercel.app/health
   curl https://your-app.vercel.app/model-info
   ```

2. **Update Frontend URLs**
   - Update `API_BASE_URL` in `simple_presentation_app.py`
   - Test predictions through web interface

3. **Monitor Performance**
   - Check Vercel dashboard for function logs
   - Monitor response times and error rates
   - Optimize based on usage patterns

## 📸 Demo Script

Once deployed, demonstrate:

1. **Live API**: Show working endpoints
2. **Web Interface**: Upload images for prediction
3. **Retraining**: Upload ZIP files and trigger retraining
4. **Load Testing**: Run Locust against live URL
5. **Mobile Access**: Show responsive design

Your malaria detection system will be live and accessible globally! 🌍
