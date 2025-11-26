# Video Playback Features

This document describes the video frame-by-frame playback and camera motion visualization features added to Depth Anything 3.

## Overview

When you process a video file, the system now:
1. **Extracts per-frame point clouds** - Each frame is stored separately with its own point cloud
2. **Captures camera poses** - Camera position and orientation for each frame (4x4 transformation matrix)
3. **Provides playback API** - Access individual frames or sequences for smooth playback
4. **Enables camera path visualization** - Show the camera's motion through 3D space

## API Endpoints

### 1. Get Video Information
```bash
GET /api/video/info
```

Returns metadata about the currently loaded video sequence.

**Response:**
```json
{
  "num_frames": 20,
  "fps": 15,
  "metadata": {
    "filename": "video.mp4",
    "resolution": 504,
    "total_points": 1000000,
    "points_per_frame": [50000, 50000, ...]
  }
}
```

### 2. Get Single Frame
```bash
GET /api/video/frame/<frame_index>
```

Returns a specific frame's point cloud data.

**Parameters:**
- `frame_index`: Frame number (0-indexed)

**Response:**
```json
{
  "frame_index": 0,
  "vertices": [[x, y, z], ...],
  "colors": [[r, g, b], ...],
  "camera_pose": [[4x4 matrix]],
  "intrinsics": [[3x3 matrix]],
  "num_points": 50000,
  "confidence": [0.9, 0.8, ...] // optional
}
```

### 3. Get Frame Range
```bash
GET /api/video/frames?start=0&end=10
```

Returns multiple frames for batch loading.

**Query Parameters:**
- `start` (optional): Starting frame index (default: 0)
- `end` (optional): Ending frame index (default: num_frames)

**Response:**
```json
{
  "frames": [
    { /* frame 0 data */ },
    { /* frame 1 data */ },
    ...
  ],
  "start": 0,
  "end": 10,
  "total": 20
}
```

### 4. Get Camera Path
```bash
GET /api/video/camera_path
```

Returns camera positions and poses for path visualization.

**Response:**
```json
{
  "poses": [
    [[4x4 transformation matrix], ...],
  ],
  "positions": [
    [x, y, z],  // Camera position for each frame
    ...
  ],
  "num_frames": 20
}
```

## Video Processing Parameters

When uploading a video via `/api/process`, you can control frame extraction:

```javascript
const formData = new FormData();
formData.append('file', videoFile);
formData.append('resolution', '504');           // Processing resolution
formData.append('max_points', '1000000');       // Total points across all frames
formData.append('feat_vis_fps', '15');          // Target FPS for playback
formData.append('show_cameras', 'true');        // Include camera poses
formData.append('include_confidence', 'true');  // Include confidence scores
```

## Frontend Implementation Examples

### Example 1: Simple Frame-by-Frame Playback

```javascript
class VideoPlayer {
    constructor() {
        this.currentFrame = 0;
        this.videoInfo = null;
        this.isPlaying = false;
        this.pointCloud = null; // Three.js Points object
    }

    async loadVideo() {
        // Get video info
        const res = await fetch('/api/video/info');
        this.videoInfo = await res.json();

        // Load first frame
        await this.loadFrame(0);
    }

    async loadFrame(frameIndex) {
        const res = await fetch(`/api/video/frame/${frameIndex}`);
        const frame = await res.json();

        // Update Three.js point cloud
        this.updatePointCloud(frame.vertices, frame.colors);
        this.currentFrame = frameIndex;
    }

    updatePointCloud(vertices, colors) {
        // Convert to Three.js format
        const positions = new Float32Array(vertices.flat());
        const colorsArray = new Float32Array(colors.flat());

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));

        if (this.pointCloud) {
            scene.remove(this.pointCloud);
        }

        const material = new THREE.PointsMaterial({
            size: 0.01,
            vertexColors: true
        });
        this.pointCloud = new THREE.Points(geometry, material);
        scene.add(this.pointCloud);
    }

    async play() {
        if (!this.videoInfo) return;

        this.isPlaying = true;
        const frameDelay = 1000 / this.videoInfo.fps;

        const playFrame = async () => {
            if (!this.isPlaying) return;

            await this.loadFrame(this.currentFrame);
            this.currentFrame = (this.currentFrame + 1) % this.videoInfo.num_frames;

            setTimeout(playFrame, frameDelay);
        };

        playFrame();
    }

    pause() {
        this.isPlaying = false;
    }

    stop() {
        this.pause();
        this.currentFrame = 0;
        this.loadFrame(0);
    }
}

// Usage
const player = new VideoPlayer();
await player.loadVideo();
player.play();
```

### Example 2: Camera Path Visualization

```javascript
async function visualizeCameraPath() {
    const res = await fetch('/api/video/camera_path');
    const data = await res.json();

    // Create line geometry for camera path
    const pathPoints = data.positions.map(pos => new THREE.Vector3(...pos));
    const pathGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    const pathMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
    const pathLine = new THREE.Line(pathGeometry, pathMaterial);
    scene.add(pathLine);

    // Add camera frustum visualizations
    data.poses.forEach((pose, index) => {
        const frustum = createCameraFrustum(pose);
        scene.add(frustum);
    });
}

function createCameraFrustum(poseMatrix) {
    // Create a small pyramid to represent camera
    const geometry = new THREE.ConeGeometry(0.05, 0.1, 4);
    geometry.rotateX(Math.PI / 2); // Point along Z axis

    const material = new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        wireframe: true
    });
    const frustum = new THREE.Mesh(geometry, material);

    // Apply camera pose
    const matrix = new THREE.Matrix4();
    matrix.fromArray(poseMatrix.flat());
    frustum.applyMatrix4(matrix);

    return frustum;
}
```

### Example 3: Looping Playback with Buffering

```javascript
class BufferedVideoPlayer extends VideoPlayer {
    constructor() {
        super();
        this.frameBuffer = new Map();
        this.bufferSize = 5; // Pre-load 5 frames
    }

    async preloadFrames(startIndex) {
        const endIndex = Math.min(
            startIndex + this.bufferSize,
            this.videoInfo.num_frames
        );

        // Load range of frames
        const res = await fetch(
            `/api/video/frames?start=${startIndex}&end=${endIndex}`
        );
        const data = await res.json();

        // Store in buffer
        data.frames.forEach(frame => {
            this.frameBuffer.set(frame.frame_index, frame);
        });
    }

    async loadFrame(frameIndex) {
        // Check if frame is in buffer
        if (!this.frameBuffer.has(frameIndex)) {
            await this.preloadFrames(frameIndex);
        }

        const frame = this.frameBuffer.get(frameIndex);
        this.updatePointCloud(frame.vertices, frame.colors);

        // Preload next batch if needed
        if ((frameIndex + 1) % this.bufferSize === 0) {
            this.preloadFrames(frameIndex + 1);
        }
    }
}
```

## Data Structure Reference

### Camera Pose Matrix (4x4)

The camera pose is a 4x4 transformation matrix in the format:

```
[
  [R00, R01, R02, Tx],  // Row 0: Rotation + X translation
  [R10, R11, R12, Ty],  // Row 1: Rotation + Y translation
  [R20, R21, R22, Tz],  // Row 2: Rotation + Z translation
  [0,   0,   0,   1 ]   // Row 3: Homogeneous coordinate
]
```

Where:
- `R` = 3x3 rotation matrix
- `T` = [Tx, Ty, Tz] translation vector (camera position)

### Camera Intrinsics Matrix (3x3)

```
[
  [fx,  0, cx],  // fx = focal length X, cx = principal point X
  [ 0, fy, cy],  // fy = focal length Y, cy = principal point Y
  [ 0,  0,  1]   // Homogeneous coordinate
]
```

## Performance Tips

1. **Pre-buffer frames** - Load multiple frames ahead of time for smooth playback
2. **Use lower max_points** - Reduce total points for faster frame loading
3. **Adjust feat_vis_fps** - Lower FPS for smoother playback on slower connections
4. **Cache frames client-side** - Store frames in memory for loop playback
5. **Use range queries** - Load frames in batches with `/api/video/frames?start=X&end=Y`

## Integration with Existing UI

To add video playback to the existing index.html:

1. **Detect video in metadata**:
```javascript
if (pointCloud.metadata.is_video) {
    showVideoControls();
}
```

2. **Add playback controls** (play, pause, stop, frame slider):
```html
<div id="video-controls" style="display:none;">
    <button onclick="videoPlayer.play()">▶ Play</button>
    <button onclick="videoPlayer.pause()">⏸ Pause</button>
    <button onclick="videoPlayer.stop()">⏹ Stop</button>
    <input type="range" id="frame-slider"
           min="0" max="0" value="0"
           oninput="videoPlayer.loadFrame(this.value)">
    <span id="frame-counter">0 / 0</span>
</div>
```

3. **Toggle camera path visualization**:
```html
<button onclick="toggleCameraPath()">
    📹 Show Camera Path
</button>
```

## Example Workflow

1. **Upload video**:
```bash
curl -X POST http://localhost:5000/api/process \
  -F "file=@video.mp4" \
  -F "max_points=1000000" \
  -F "feat_vis_fps=15" \
  -F "show_cameras=true"
```

2. **Get video info**:
```bash
curl http://localhost:5000/api/video/info
```

3. **Play frames in loop**:
```javascript
// Load frames 0-19 in sequence, then repeat
for (let i = 0; ; i = (i + 1) % 20) {
    const frame = await fetch(`/api/video/frame/${i}`).then(r => r.json());
    displayFrame(frame);
    await sleep(1000/15); // 15 FPS
}
```

4. **Show camera path**:
```bash
curl http://localhost:5000/api/video/camera_path
```

## Future Enhancements

- [ ] WebSocket streaming for real-time playback
- [ ] Variable speed playback
- [ ] Frame interpolation for smoother playback
- [ ] Export video sequence as animated GLB/GLTF
- [ ] Camera path smoothing and filtering
- [ ] Multi-view synchronization (split screen)

## See Also

- [app.py](app.py) - Backend implementation
- [index.html](docs/index.html) - Frontend viewer
- [README.md](README.md) - Main documentation
