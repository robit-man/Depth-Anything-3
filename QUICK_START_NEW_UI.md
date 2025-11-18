# 🚀 Quick Start - New Full-Screen UI

## What's New?

You now have a **completely redesigned interface** with:
- ✅ Full-screen Three.js point cloud viewer
- ✅ Model selection modal (7 models to choose from)
- ✅ Drag-and-drop file upload
- ✅ Automatic floor detection and alignment
- ✅ Professional glass-morphism UI
- ✅ Real-time status indicators

## Start the New Interface

```bash
python3 main.py
```

The server will start on port 5000 (or next available port if 5000 is in use).

## First Time Usage

### 1. Open the Interface
Open your browser to: `http://localhost:5000`

You'll see:
- Black Three.js scene with grid and axes
- Top bar with model selector and file upload
- Bottom bar with export and floor alignment (disabled until you process a file)
- Status indicator in top-right showing "Model: Not loaded"

### 2. Select a Model
Click **"Select Model"** in the top bar to open the model selection modal.

**Available Models:**
- **DA3 Small** (~2GB) - Fastest, real-time applications
- **DA3 Base** (~3GB) - Balanced for general use
- **DA3 Large** (~4GB) - High quality depth estimation
- **DA3 Giant** (~5GB) - Maximum quality
- **DA3 Nested Giant+Large** (~8GB) - Default, best results with metric depth ⭐
- **DA3 Metric Large** (~4GB) - Metric depth applications
- **DA3 Mono Large** (~4GB) - Single camera specialization

Each model card shows:
- ✅ Download status (already downloaded or needs download)
- Size, speed, and quality ratings
- Recommended use case
- Current model has a green checkmark

### 3. Load the Model
After selecting a model, click **"Load Model"** in the top bar.

You'll see:
- Status changes to "Loading..."
- Progress bar appears
- Model downloads from Hugging Face (first time only)
- Status changes to "Ready" when complete

⏱️ **First download takes 5-10 minutes** depending on your connection.
💡 **Subsequent loads are instant** (models are cached).

### 4. Upload a File

**Option A: Drag and Drop** (Recommended)
- Drag any image (JPG, PNG) or video (MP4, AVI, MOV) over the window
- A beautiful overlay appears
- Drop the file
- Processing starts automatically

**Option B: File Browser**
- Click **"Browse Files"** in the top bar
- Select an image or video
- Processing starts automatically

### 5. View Your Point Cloud
Once processing completes:
- Point cloud appears in the 3D scene
- Use mouse to interact:
  - **Left drag**: Rotate view
  - **Right drag**: Pan
  - **Scroll**: Zoom in/out
- Status shows point count
- Bottom buttons become enabled

### 6. Align the Floor (Optional)
If your point cloud has a floor/ground plane:
1. Click **"Align with Floor"** in the bottom bar
2. RANSAC algorithm automatically detects the floor plane
3. Point cloud rotates and translates to align floor with y=0
4. The floor now sits on the grid

The algorithm:
- Finds the lowest 5% of points (floor candidates)
- Runs 50 RANSAC iterations to find the dominant plane
- Ensures the plane is roughly horizontal
- Rotates the entire point cloud to align the floor with the Y-axis
- Translates everything so the floor is at y=0

### 7. Export Your Work
Click **"Export GLB"** in the bottom bar to download your point cloud as a GLB file.

You can then import it into:
- Blender
- Unity
- Unreal Engine
- Any 3D software that supports GLB/GLTF

---

## Tips & Tricks

### Switching Models
1. Click "Select Model" anytime
2. Choose a different model
3. Click "Load Model" to switch
4. **Downloaded models load instantly**
5. New models download first (5-10 minutes)

### Using Smaller Models for Speed
If you need faster processing:
1. Select "DA3 Small" (fastest)
2. Resolution will be lower but processing is 3-4x faster
3. Great for testing or real-time applications

### Using Larger Models for Quality
For maximum quality:
1. Select "DA3 Giant" or "DA3 Nested Giant+Large"
2. Processing takes longer but results are stunning
3. Recommended for final production renders

### Processing Videos
1. Upload a video file (MP4, AVI, MOV)
2. Backend extracts frames automatically
3. Each frame is processed
4. Point clouds are combined
5. Result is a complete 3D reconstruction

### Keyboard Shortcuts
- **Double-click** model card to select and load immediately
- **Escape** to close model selection modal
- **R** to reset view (when point cloud is loaded)

### Performance Tips
- **Resolution**: Default is 504. Higher = better quality but slower
- **Max Points**: Default is 1M. Lower = faster rendering
- **Model Size**: Smaller models = faster processing
- **GPU**: CUDA greatly speeds up inference

---

## Troubleshooting

### Model Won't Load
- Check internet connection (models download from Hugging Face)
- Check disk space (~2-8GB per model)
- Wait for download to complete (check terminal for progress)
- Try a smaller model first (DA3 Small)

### Point Cloud Doesn't Appear
- Check status indicator - is model ready?
- Check browser console for errors (F12)
- Try reloading the page
- Check that file is valid image/video

### Floor Alignment Doesn't Work
- Ensure point cloud has a clear floor/ground plane
- Camera should be facing generally forward (not straight down)
- Floor should be visible in the original image
- Try adjusting the confidence threshold (advanced users)

### UI Looks Broken
- Try different browser (Chrome/Firefox recommended)
- Check browser console for errors
- Ensure JavaScript is enabled
- Clear browser cache and reload

---

## Advanced Usage

### API Integration
The new UI preserves **all existing RESTful API endpoints**!

You can still use:
```bash
# Process image via API
curl -X POST http://localhost:5000/api/v1/process/image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'

# List available models
curl http://localhost:5000/api/models/list

# Select and load a model
curl -X POST http://localhost:5000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{"model_id": "da3-small"}'

curl -X POST http://localhost:5000/api/load_model
```

See [API_README.md](API_README.md) for full API documentation.

### Custom Configurations
Edit `main.py` to customize:
- `MAX_CONTENT_LENGTH` (line 250) - Max file size
- `UPLOAD_FOLDER` / `OUTPUT_FOLDER` (lines 251-252) - Directories
- Floor detection parameters (line 298) - Adjust sensitivity
- Three.js scene settings (lines 1068-1106) - Lighting, colors, etc.

---

## What's Preserved from Old Version?

✅ **API Endpoints**: All RESTful API routes still work
✅ **Bootstrap**: venv setup unchanged
✅ **Dependencies**: Same requirements
✅ **Processing**: Same DA3 inference pipeline
✅ **Export**: GLB export compatible

**Backup Available**: [main_backup_v1.py](main_backup_v1.py)

---

## File Structure

```
depth-anything-3/
├── main.py                     # ⭐ New full-screen UI
├── main_backup_v1.py          # Backup of old version
├── api_endpoints.py           # RESTful API (unchanged)
├── example_api_client.py      # API usage examples
├── example_threejs_client.html # Alternative Three.js client
├── quick_api_demo.py          # API demo script
├── UI_REDESIGN_SUMMARY.md     # ⭐ Complete redesign documentation
├── QUICK_START_NEW_UI.md      # ⭐ This file
├── API_README.md              # API documentation
└── API_IMPLEMENTATION_PLAN.md # API technical spec
```

---

## Support

- **Documentation**: See [UI_REDESIGN_SUMMARY.md](UI_REDESIGN_SUMMARY.md) for complete details
- **API Guide**: See [API_README.md](API_README.md) for API usage
- **Issues**: Check GitHub issues at https://github.com/ByteDance-Seed/Depth-Anything-3/issues
- **Original Docs**: See [README.md](README.md) for DA3 documentation

---

## Next Steps

1. ✅ Start the server: `python3 main.py`
2. ✅ Open browser to `http://localhost:5000`
3. ✅ Select and load a model
4. ✅ Upload an image or video
5. ✅ Explore your 3D point cloud
6. ✅ Try floor alignment
7. ✅ Export as GLB

**Enjoy your new professional 3D point cloud studio!** 🎉
