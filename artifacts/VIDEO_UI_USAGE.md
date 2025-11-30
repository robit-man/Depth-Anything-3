# Video Playback UI - User Guide

## Quick Start

1. **Upload a video** through the web interface (drag-and-drop or file upload)
2. **Wait for processing** - the API will extract frames and camera poses
3. **Video controls appear automatically** in the bottom bar when a video is loaded
4. **Use the playback controls** to navigate through frames

## Video Controls

The video control panel appears in the bottom bar when a video is detected:

```
┌────────────────────────────────────────────────────┐
│  ▶  ⏸  ⏹  🔁  📹        Frame: 5 / 20             │
│  ══════════════════════════════════════  15 FPS    │
└────────────────────────────────────────────────────┘
```

### Control Buttons

| Button | Icon | Function | Keyboard |
|--------|------|----------|----------|
| **Play** | ▶ | Start frame-by-frame playback | - |
| **Pause** | ⏸ | Pause playback | - |
| **Stop** | ⏹ | Stop and return to frame 0 | - |
| **Loop** | 🔁 | Toggle loop mode (on by default) | - |
| **Camera Path** | 📹 | Show/hide camera motion path | - |

### Timeline Slider

- **Drag** the slider to seek to any frame
- **Current frame** and **total frames** are displayed
- **FPS** (frames per second) is shown on the right

## Features

### 1. Frame-by-Frame Playback

The video plays back as a sequence of point clouds:
- Each frame is a complete 3D point cloud
- Smooth transitions between frames
- Configurable playback speed (based on video FPS)

**How it works:**
- Frames are buffered in batches of 5 for smooth playback
- Point clouds are swapped out in real-time
- Loop mode automatically restarts from frame 0

### 2. Loop Mode (Default: ON)

When enabled (button highlighted):
- Playback automatically restarts from frame 0 after the last frame
- Perfect for continuous viewing
- Disabled: playback stops at the last frame

### 3. Camera Path Visualization

Toggle the camera path button to see:

**Red Path Line:**
- Shows the camera's movement through 3D space
- Connects all camera positions in sequence

**Green Camera Frustums:**
- Small pyramid wireframes at each frame position
- Shows camera orientation at each point
- Can be clicked to see frame index (stored in userData)

### 4. Manual Frame Seeking

Use the slider to jump to any frame:
- Drag to desired position
- Playback pauses during seeking
- If playing before seek, resumes after

## Workflow Examples

### Example 1: Review Camera Motion

1. Upload video of room/scene
2. Wait for processing (API extracts ~20 frames)
3. Click **Camera Path** button (📹)
4. Observe the red line showing camera movement
5. Green frustums show where camera was pointing
6. Use slider to inspect specific positions

### Example 2: Create Looping Animation

1. Upload video
2. Click **Play** (▶)
3. Loop mode is ON by default (🔁 highlighted)
4. Video plays continuously in a loop
5. Good for demos or presentations

### Example 3: Compare Specific Frames

1. Upload video
2. Use slider to go to frame 5
3. Note details in point cloud
4. Drag slider to frame 15
5. Compare the two frames
6. Use ◀/▶ for fine adjustments

## Advanced Tips

### Performance Optimization

**For Smooth Playback:**
- Use lower resolution (504 instead of 672)
- Reduce max_points (500k instead of 1M)
- Shorter videos process faster (API caps at 20 frames)

**Upload Settings:**
```javascript
// In config modal before upload:
Resolution: 504
Max Points: 500000
FPS: 15 (smooth 15fps playback)
Show Cameras: ✓ (required for camera path)
```

### Camera Path Details

The camera path visualization shows:
- **Position** (X, Y, Z coordinates) of camera at each frame
- **Orientation** (rotation) shown by frustum direction
- **Path continuity** - smooth motion = smooth line, jumpy = scattered points

**Use Cases:**
- Verify camera tracking quality
- Understand scene coverage
- Identify areas the camera focused on
- Debug depth estimation artifacts

### Keyboard Shortcuts (Future Enhancement)

Planned shortcuts:
- `Space` - Play/Pause
- `Left/Right` - Previous/Next frame
- `Home` - First frame
- `End` - Last frame
- `L` - Toggle loop
- `C` - Toggle camera path

## Technical Details

### Frame Buffer
- Preloads 5 frames ahead
- Reduces network latency
- Enables smooth playback

### Point Cloud Swapping
- Each frame is a separate point cloud
- Old cloud removed before new one loads
- Memory-efficient (only active frame in GPU)

### API Endpoints Used
- `GET /api/video/info` - Get video metadata
- `GET /api/video/frame/<index>` - Load specific frame
- `GET /api/video/frames?start=X&end=Y` - Batch load frames
- `GET /api/video/camera_path` - Get camera positions/poses

## Troubleshooting

### Video controls don't appear
**Cause:** Video metadata not detected
**Fix:**
1. Check console for errors
2. Verify `show_cameras=true` was set during upload
3. Re-upload video with correct settings

### Playback is choppy
**Causes:**
- Too many points per frame
- Slow network connection
- Browser GPU limitations

**Fixes:**
- Reduce max_points in upload config
- Use lower resolution (504)
- Close other browser tabs
- Check network connection

### Camera path not visible
**Causes:**
- Camera path not toggled on
- Cameras are outside current view
- All cameras at same position

**Fixes:**
- Click camera path button (📹)
- Reset view to see full scene
- Check if video had camera motion

### Frames load slowly
**Causes:**
- Large point clouds
- Network latency
- Server processing

**Fixes:**
- Wait for buffer to fill (first few frames slower)
- Reduce max_points setting
- Pause and let frames preload

## Browser Compatibility

**Tested and Working:**
- Chrome/Edge 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓

**Requirements:**
- WebGL 2.0 support
- ES6+ JavaScript
- Fetch API

**Not Supported:**
- Internet Explorer (any version)
- Old mobile browsers

## See Also

- [VIDEO_PLAYBACK.md](VIDEO_PLAYBACK.md) - API documentation
- [README.md](README.md) - Main documentation
- [app.py](app.py) - Backend implementation
- [index.html](docs/index.html) - Frontend code
