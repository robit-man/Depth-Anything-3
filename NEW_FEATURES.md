# New Features Added

## 1. ✅ Video File Support

### Problem
Only image files were being processed. Video files (MP4, AVI, MOV, etc.) were rejected or caused errors.

### Solution
Added automatic video detection and frame extraction:

```python
# Detect video files by extension
file_ext = filepath.suffix.lower()
is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

if is_video:
    # Extract frames using moviepy
    clip = VideoFileClip(str(filepath))

    # Sample frames every 0.5 seconds
    sample_rate = max(1, int(fps * 0.5))

    # Limit to 20 frames max for performance
    for i, frame in enumerate(clip.iter_frames()):
        if i % sample_rate == 0:
            frames.append(Image.fromarray(frame))
        if len(frames) >= 20:
            break

    # Process all frames together
    prediction = state.model.inference(
        image=frames,
        process_res=resolution,
        num_max_points=max_points
    )
```

### Supported Video Formats
- ✅ `.mp4` - MPEG-4
- ✅ `.avi` - Audio Video Interleave
- ✅ `.mov` - QuickTime
- ✅ `.mkv` - Matroska
- ✅ `.webm` - WebM
- ✅ `.flv` - Flash Video

### Features
- **Automatic frame extraction**: Samples 1 frame every 0.5 seconds
- **Performance limit**: Max 20 frames processed per video
- **Combined point cloud**: All frames merged into single 3D reconstruction
- **Same interface**: Works with both drag-and-drop and file browser

### Usage
1. Drag and drop a video file onto the page
2. Or click "Browse Files" and select a video
3. System automatically detects it's a video
4. Frames are extracted and processed
5. Point cloud appears showing the scene from multiple views

---

## 2. ✅ First-Person Navigation (Walk Mode)

### Problem
Could only orbit around the scene. No way to walk through the point cloud at eye level.

### Solution
Added keyboard controls for first-person navigation:

```javascript
function setupKeyboardControls() {
    const moveSpeed = 0.1;   // Movement speed (meters per frame)
    const eyeLevel = 1.6;    // Eye level height (1.6 meters)

    document.addEventListener('keydown', (e) => {
        // Get camera forward direction
        const direction = new THREE.Vector3();
        camera.getWorldDirection(direction);
        direction.y = 0;  // Keep horizontal
        direction.normalize();

        // Calculate right vector
        const right = new THREE.Vector3();
        right.crossVectors(direction, new THREE.Vector3(0, 1, 0));

        // Move based on key pressed
        switch(e.key) {
            case 'W': camera.position.add(direction * moveSpeed); break;
            case 'S': camera.position.add(direction * -moveSpeed); break;
            case 'A': camera.position.add(right * -moveSpeed); break;
            case 'D': camera.position.add(right * moveSpeed); break;
        }

        // Maintain eye level
        camera.position.y = eyeLevel;
    });
}
```

### Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **W** or **↑** | Move Forward | Walk in direction camera is facing |
| **S** or **↓** | Move Backward | Walk backward |
| **A** or **←** | Strafe Left | Move left (perpendicular to view) |
| **D** or **→** | Strafe Right | Move right (perpendicular to view) |
| **Q** | Move Up | Fly upward (vertical) |
| **E** | Move Down | Fly downward (vertical) |
| **Space** | Reset Height | Return to eye level (1.6m) |

### Features
- **Eye-level navigation**: Camera stays at realistic 1.6m height
- **Direction-based movement**: Forward/backward follows where you're looking
- **Strafing**: Left/right movement perpendicular to view direction
- **Smooth movement**: 0.1m per keypress (adjustable)
- **Vertical control**: Q/E for flying up/down
- **Height reset**: Space bar returns to eye level

### Physics
- **Eye level**: 1.6 meters (average human eye height)
- **Move speed**: 0.1 meters per frame (about 6 km/h walking speed)
- **Direction vector**: Calculated from camera's world direction
- **Right vector**: Cross product of direction and up vector

### Combined with Mouse
- **Mouse drag**: Still rotates view (look around)
- **Keyboard**: Moves position (walk around)
- **Together**: Full first-person control like a game

### Usage Examples

**Walk through a room:**
1. Process an image/video of an indoor space
2. Use mouse to look around
3. Press W to walk forward
4. Press A/D to strafe around objects
5. Press Q/E to change height if needed

**Explore a landscape:**
1. Process outdoor scene
2. Use W/S to walk forward/backward
3. Mouse to change view direction
4. Space to reset to ground level

**Inspect details:**
1. Walk close to interesting areas with WASD
2. Use mouse to look at different angles
3. Press Q to rise up for overview
4. Press E to crouch down for low angle

---

## 3. ✅ Eye-Level Defaults

### Changes
- **Initial camera position**: `(0, 1.6, 5)` instead of `(0, 2, 5)`
- **Reset view height**: 1.6m instead of 2m
- **Look-at target**: Eye level `(0, 1.6, 0)` instead of ground `(0, 0, 0)`

### Why 1.6 meters?
- Average human eye height when standing
- Provides realistic perspective
- Standard in VR/AR applications
- Natural for architectural visualization

---

## Testing Guide

### Test Video Support

```bash
# 1. Start server
python3 main.py

# 2. Open browser to http://localhost:5000

# 3. Load model

# 4. Test video upload
#    - Drag and drop an MP4 file
#    - OR click "Browse Files" and select video
#    - Watch console for frame extraction logs
#    - Point cloud should appear with multiple viewpoints merged
```

**Expected Console Output:**
```
Processing video file...
Extracting frames (FPS: 30, Duration: 10s)...
Extracted 20 frames
Processing frames with DA3...
Point cloud generated with X points
```

### Test Keyboard Navigation

```bash
# 1. After loading a point cloud:

# 2. Press W (or ↑)
#    - Camera moves forward
#    - Point cloud gets closer
#    - Console: "Keyboard controls enabled: WASD/Arrows=Move..."

# 3. Press A/D (or ←/→)
#    - Camera strafes left/right
#    - Point cloud shifts sideways

# 4. Press S (or ↓)
#    - Camera moves backward
#    - Point cloud gets farther

# 5. Use mouse to rotate view
#    - Change where W points
#    - Forward direction follows view

# 6. Press Q/E
#    - Camera rises/falls
#    - Height changes

# 7. Press Space
#    - Height resets to 1.6m
#    - Returns to eye level
```

### Test Combined Navigation

```bash
# Walk in a circle around point cloud:
1. Mouse drag right → View rotates right
2. Press W → Walk forward (in new direction)
3. Mouse drag right → View rotates more
4. Press W → Walk forward again
5. Repeat → Circle around scene

# Explore from different heights:
1. Press Q 10 times → Rise up (bird's eye view)
2. Mouse look down → See point cloud from above
3. Press Space → Return to eye level
4. Press E 10 times → Lower down (ground view)
5. Mouse look up → See point cloud from below
```

---

## Implementation Details

### Video Processing Flow

```
Video file uploaded
    ↓
Detect video by extension
    ↓
Open with moviepy.VideoFileClip
    ↓
Calculate sample rate (FPS * 0.5)
    ↓
Extract frames (max 20)
    ↓
Convert frames to PIL Images
    ↓
Pass all frames to DA3 inference
    ↓
Merge point clouds from all frames
    ↓
Display combined 3D reconstruction
```

### Keyboard Control Flow

```
User presses key
    ↓
Get camera forward direction
    ↓
Project to horizontal plane (y=0)
    ↓
Normalize direction vector
    ↓
Calculate right vector (cross product)
    ↓
Move camera position based on key:
  - W/↑: position += direction * speed
  - S/↓: position -= direction * speed
  - A/←: position -= right * speed
  - D/→: position += right * speed
  - Q: position.y += speed
  - E: position.y -= speed
    ↓
Clamp height to eye level (except Q/E)
    ↓
Render new view
```

---

## Configuration

### Adjustable Parameters

**Video Processing:**
```python
# In main.py, line ~539
sample_rate = max(1, int(fps * 0.5))  # Change 0.5 to adjust frequency
frames_limit = 20  # Change max frames processed
```

**Keyboard Movement:**
```javascript
// In main.py, line ~1241
const moveSpeed = 0.1;  // Increase for faster movement
const eyeLevel = 1.6;   // Adjust eye height
```

### Performance Tips

**For faster video processing:**
- Increase `sample_rate` (e.g., `fps * 1.0` for every 1 second)
- Decrease `frames_limit` (e.g., `10` for max 10 frames)
- Use lower resolution videos

**For smoother navigation:**
- Decrease `moveSpeed` (e.g., `0.05` for slower, more precise)
- Increase `moveSpeed` (e.g., `0.2` for faster exploration)
- Adjust `eyeLevel` for different perspectives

---

## Known Limitations

### Video Processing
- **Max 20 frames**: Prevents memory issues with long videos
- **Sample rate**: Only processes every 0.5 seconds (not every frame)
- **Large files**: Very long/high-res videos may take time to extract frames
- **Codec support**: Depends on moviepy/ffmpeg capabilities

### Keyboard Navigation
- **No collision detection**: Can walk through point clouds
- **No gravity**: Can fly freely with Q/E
- **Fixed speed**: Movement speed is constant (no acceleration)
- **No inertia**: Instant stop when key released

---

## Future Enhancements

Potential improvements:
- [ ] Adjustable movement speed slider
- [ ] Collision detection with point cloud
- [ ] Gravity/ground snapping
- [ ] Sprint mode (Shift key)
- [ ] Crouch mode (Ctrl key)
- [ ] Smooth acceleration/deceleration
- [ ] Gamepad support
- [ ] VR headset support
- [ ] Video timeline scrubber
- [ ] Frame-by-frame video navigation
- [ ] Multi-video merging
- [ ] Real-time video streaming

---

## Summary

### Video Support
✅ Drag and drop MP4, AVI, MOV, etc.
✅ Automatic frame extraction
✅ Combined 3D reconstruction
✅ Same UI as images

### Keyboard Navigation
✅ WASD + Arrow keys for movement
✅ Q/E for vertical
✅ Space for height reset
✅ Eye-level perspective (1.6m)
✅ Direction-based forward/back
✅ Strafing left/right

**Both features work together for immersive exploration of generated 3D scenes!** 🎮
