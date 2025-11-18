# 🚀 Depth Anything 3 - Flask Application Quick Start

## One-Command Launch

Simply run from the base directory:

```bash
python3 main.py
```

That's it! The application will automatically:
- ✅ Create a virtual environment (`venv/`)
- ✅ Install all dependencies from `requirements.txt`
- ✅ Install the depth-anything-3 package
- ✅ Install Flask and Flask-CORS
- ✅ Start the web server

## First Run (Automatic Setup)

On the first run, you'll see:

```
======================================================================
🚀 Depth Anything 3 - Flask Application Bootstrap
======================================================================

📦 Creating virtual environment...
✓ Virtual environment created

📥 Installing dependencies from requirements.txt...
⏳ This may take several minutes...

📦 Installing PyTorch (this is a large package)...
📦 Installing other dependencies...
📦 Installing depth-anything-3 package...
📦 Installing Flask...
✓ All dependencies installed

🔄 Restarting in virtual environment...
```

The application will then automatically restart within the virtual environment and start the Flask server.

## Subsequent Runs

After the first setup, launching is instant:

```bash
python3 main.py
```

You'll see:

```
======================================================================
🚀 Depth Anything 3 - Flask Application
======================================================================
✓ Virtual environment: Active
✓ Dependencies: Installed
✓ CUDA Available: True/False
======================================================================

📡 Starting Flask server...
🌐 Open your browser and navigate to: http://localhost:5000
```

## Using the Application

### Step 1: Access the Interface

Open your browser and go to: **http://localhost:5000**

### Step 2: Load the Model

1. Click the **"Load Model"** button
2. Wait for the model to download and load (first time only, ~5GB)
3. Model status will change from "Loading..." to "Ready"

### Step 3: Upload Video/Images

1. **Drag & drop** a video file onto the upload area, OR
2. **Click** the upload area to select a file
3. Supported formats: MP4, AVI, MOV, PNG, JPG

### Step 4: Configure Settings

Adjust processing parameters:

- **Video FPS**: Frames to extract per second (higher = more frames, slower processing)
- **Processing Resolution**: Quality vs. speed tradeoff
  - 504px: Fast processing
  - 756px: Balanced
  - 1024px: High quality
- **Export Format**: Output file types
  - GLB: 3D point cloud only
  - GLB + Depth Visualization: Point cloud + depth maps
  - NPZ + GLB: Numerical data + 3D visualization
- **Max Points**: Maximum points in the point cloud (lower = faster, higher = more detail)

### Step 5: Process

1. Click **"Start Processing"**
2. Watch the progress bar and status messages
3. Processing time depends on:
   - Video length
   - FPS setting
   - Resolution
   - GPU availability

### Step 6: View Results

- The 3D point cloud automatically loads in the viewer
- **Controls:**
  - 🖱️ Left-click + drag: Rotate
  - 🖱️ Right-click + drag: Pan
  - 🔄 Scroll wheel: Zoom
  - 📷 "Reset View" button: Return to default camera position

## Features

### 🎯 Single-Page Interface
Everything in one place:
- Model status monitoring
- File upload with drag & drop
- Real-time processing progress
- Interactive 3D viewer
- Activity logs

### 🧠 Smart Model Management
- Automatic model downloading from Hugging Face
- Progress tracking during download
- GPU detection and usage
- Model persistence (loads once per session)

### 📊 Real-Time Progress
- Model loading progress
- Frame extraction status
- Processing percentage
- Detailed activity logs

### 🎨 3D Point Cloud Viewer
- Three.js-powered interactive viewer
- Orbit controls (rotate, pan, zoom)
- Automatic model centering and scaling
- Grid and axes helpers
- Multiple lighting sources

### ⚙️ Flexible Processing Options
- Video frame extraction with configurable FPS
- Multiple resolution options
- Various export formats
- Customizable point cloud density

## Advanced Usage

### Batch Processing

Process multiple videos sequentially by:
1. Processing first video
2. Waiting for completion
3. Uploading next video
4. Repeat

The model stays loaded between jobs for faster processing.

### Output Files

All outputs are saved in `outputs/<job_id>/`:
- `scene.glb` - 3D point cloud (loadable in Blender, Three.js viewers)
- `scene.jpg` - Preview image
- `depth_vis/` - Depth visualization images (if enabled)
- `frames/` - Extracted video frames

### API Endpoints

The application provides REST API endpoints:

- `GET /` - Main interface
- `GET /api/model_status` - Check model status
- `GET /api/device_info` - GPU/CPU information
- `POST /api/load_model` - Load the model
- `POST /api/process` - Start processing (multipart/form-data)
- `GET /api/status/<job_id>` - Get job status
- `GET /api/output/<job_id>/<filename>` - Download output files

## Troubleshooting

### Model Won't Load

**Issue:** Model fails to download

**Solutions:**
1. Check internet connection
2. Ensure enough disk space (~5GB for model)
3. Try using HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`

### CUDA Not Available

**Issue:** "CUDA Available: False"

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA installation
3. The app will still work on CPU (slower)

### Out of Memory

**Issue:** GPU runs out of memory

**Solutions:**
1. Reduce processing resolution (504px)
2. Lower max points (500,000)
3. Reduce video FPS (1.0)
4. Process shorter videos
5. Close other GPU-intensive applications

### Slow Processing

**Performance tips:**
1. Use GPU if available (10-100x faster)
2. Lower resolution for preview processing
3. Reduce FPS for long videos
4. Reduce max points for faster point cloud generation

## System Requirements

### Minimum
- Python 3.9-3.13
- 8GB RAM
- 10GB disk space
- CPU: Any modern processor

### Recommended
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 20GB+ disk space (for model and outputs)

## File Structure

After running:

```
Depth-Anything-3/
├── main.py                 # Standalone application
├── venv/                   # Virtual environment (auto-created)
├── uploads/                # Uploaded files (auto-created)
│   └── <job_id>/
│       └── video.mp4
├── outputs/                # Processing outputs (auto-created)
│   └── <job_id>/
│       ├── frames/
│       ├── scene.glb
│       └── scene.jpg
└── src/                    # Source code (existing)
```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Cleaning Up

To remove all processed data:

```bash
rm -rf uploads/ outputs/
```

To completely remove the virtual environment and start fresh:

```bash
rm -rf venv/
python3 main.py  # Will recreate and reinstall everything
```

## Getting Help

If you encounter issues:

1. Check the activity log in the web interface (bottom right)
2. Check the terminal output where you ran `main.py`
3. Ensure all dependencies are correctly installed
4. Try reducing processing parameters
5. Check the [main README](README.md) for model-specific information

## Credits

This standalone application integrates:
- **Depth Anything 3** - Core depth estimation model
- **Flask** - Web framework
- **Three.js** - 3D visualization
- **PyTorch** - Deep learning backend
- Original CSS/HTML from the DA3 Gradio app

Enjoy creating 3D point clouds from your videos! 🎉
