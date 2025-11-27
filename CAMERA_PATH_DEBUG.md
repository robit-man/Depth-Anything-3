# Camera Path Visualization - Debugging Guide

## Problem: All Camera Positions at (0,0,0)

### Symptoms
```
📹 Camera positions (first 3): (0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00)
❌ All camera positions are at origin! Check pose inversion.
```

---

## Root Cause

The Depth Anything 3 model is **NOT predicting camera poses**.

### Why This Happens

1. **Model doesn't have camera decoder**: Not all DA3 models include camera pose prediction
2. **Fallback to identity matrices**: When `prediction.extrinsics` is `None`, the code uses:
   ```python
   camera_poses = [np.eye(4) for _ in range(len(prediction.depth))]
   ```
3. **Identity matrix inversion**: `inv(I) = I`, so camera position `[:3, 3]` stays at `[0, 0, 0]`

---

## Solutions

### Option 1: Use Camera-Enabled Model ✅ RECOMMENDED

Models with camera prediction capability:
- `da3metric-large`
- `da3metric-giant`
- `da3nested-*` models with camera decoder

Check if your model has camera prediction:
```python
# In Python console
from depth_anything_3.api import DepthAnything3
model = DepthAnything3(model_name="da3metric-large")
print(model.model.cam_dec)  # Should show CameraDec module, not None
```

### Option 2: Provide Ground Truth Poses

If you have external camera tracking (from SLAM, ARKit, Vicon, etc.):

```python
# Your tracked camera poses (world-to-camera, 4x4 matrices)
extrinsics = np.array([...])  # Shape: (N, 4, 4)
intrinsics = np.array([...])   # Shape: (N, 3, 3)

prediction = model.inference(
    image=frames,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    ...
)
```

### Option 3: Synthetic Path (For Testing) 🆕

The backend now **automatically generates** a synthetic forward-walking path when the model doesn't predict poses:

```python
# app.py generates synthetic path:
# - Starts at origin (0, 0, 0)
# - Moves forward 0.5m per frame along -Z axis
# - Camera looks down -Z (standard camera convention)
```

This allows the visualization to work even without real camera tracking!

---

## Expected Behavior

### With Camera Prediction
```
🔍 Camera Pose Prediction Debug:
  • prediction.extrinsics exists: True
  • Number of poses: 20
  • First pose shape: (4, 4)
  • First pose is identity: True  ← First frame normalized to origin
  • First pose translation: [0. 0. 0.]

📹 Camera Path Analysis
Total keyframes: 20
First position (should be ~origin): (0.000, 0.000, 0.000)  ✅
Last position: (0.234, 0.012, 1.876)  ✅ Shows movement!
Path length: 1.89 meters (assuming 1 unit = 1 meter)
```

### With Synthetic Path
```
🔍 Camera Pose Prediction Debug:
  ⚠️  Model did NOT predict camera poses - using identity matrices
  ⚠️  This will result in all camera positions at (0,0,0)

⚠️  Generating synthetic camera path for visualization (model didn't predict poses)
  Generated 20 poses: walking forward 9.5 meters

📹 Camera Path Analysis
Total keyframes: 20
First position (should be ~origin): (0.000, 0.000, 0.000)  ✅
Last position: (0.000, 0.000, 9.500)  ✅ Synthetic straight line!
Path length: 9.50 meters (assuming 1 unit = 1 meter)
Direction from start to end: (0.00, 0.00, 1.00)  ← Pure +Z motion
```

---

## Coordinate System

### DA3 Convention
- **Extrinsics**: World-to-camera transformation (w2c)
- **Normalized**: First frame is at identity (origin, looking down -Z)
- **Camera Space**: +X right, +Y up, **-Z forward**

### Visualization (Three.js)
- **c2w matrices**: Camera-to-world (inverted extrinsics)
- **Camera position**: `c2w[:3, 3]`
- **World axes**:
  - **Red (X)**: Right
  - **Green (Y)**: Up
  - **Blue (Z)**: Forward (in world space)

### Walking Forward Example

If you walk forward with the camera:
- Camera moves in **-Z direction** (camera space)
- Path shows movement in **+Z direction** (world space, after inversion)
- Expected: `positions[0] = (0,0,0)`, `positions[N] = (0,0,distance)`

---

## Debugging Commands

### Backend (Python Console)
```bash
# Check model configuration
python3 -c "
from depth_anything_3.api import DepthAnything3
model = DepthAnything3(model_name='da3-large')
print('Has camera decoder:', model.model.cam_dec is not None)
"
```

### Frontend (Browser Console)
```javascript
// Check camera path data
fetch('/api/video/camera_path')
    .then(r => r.json())
    .then(data => {
        console.log('Positions:', data.positions);
        console.log('All at origin?', data.positions.every(p =>
            p.every(v => Math.abs(v) < 0.001)
        ));
    });

// Check video player state
console.log('Video player:', videoPlayer);
console.log('Camera spline:', videoPlayer.cameraSpline);
console.log('Path data:', videoPlayer.cameraPathData);
```

---

## Testing the Visualization

1. **Start backend with debugging**:
   ```bash
   python app.py
   # Watch for "Camera Pose Prediction Debug" output
   ```

2. **Upload a video** (any video will work)

3. **Run inference**

4. **Check backend console** for pose debugging output

5. **Open browser and toggle camera path**

6. **Expected results**:
   - With camera prediction: Smooth path following actual camera motion
   - Without camera prediction: Straight line forward (synthetic path)
   - Grid shows scale (1 unit = 1 meter)
   - Axes show orientation at origin

---

## Files Modified

### Backend (`app.py`)
- Added camera pose prediction debugging (lines 1375-1388)
- Added synthetic path generation (lines 1390-1401)
- Added camera path endpoint debugging (lines 1621-1637)

### Frontend (`docs/index.html`)
- Enhanced camera path analysis logging (lines 3670-3699)
- Added coordinate axes visualization (lines 3883-3946)
- Added reference grid (lines 3948-3956)
- Added comprehensive path statistics
- Added direction arrows along path
- Added animated progress marker

---

## Next Steps

1. ✅ **Test with current setup** - Should now show synthetic path
2. 🔄 **Try da3metric model** - For real camera pose prediction
3. 📊 **Check backend logs** - Verify what model is predicting
4. 🎥 **Test with real tracked video** - If you have ground truth poses

The visualization is now **robust and informative**, working with or without camera prediction! 🎉
