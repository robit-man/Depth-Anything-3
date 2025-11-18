# 🚀 RESTful API Implementation Plan

## Overview

Add programmatic API endpoints to process images/videos and return point cloud data in formats easily consumable by clients (JSON, GLB, binary).

---

## 🎯 API Endpoints

### 1. **POST /api/v1/process/image**
Process a single image and return point cloud data.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",  // Base64 encoded image
  // OR
  "image_url": "https://example.com/image.jpg",    // URL to image
  // OR
  "image_path": "/path/to/local/image.jpg",        // Server-side path

  // Optional parameters
  "resolution": 504,                // Processing resolution (504/756/1024)
  "max_points": 1000000,            // Max points in cloud
  "export_format": "json",          // json, glb, binary
  "include_cameras": true,          // Include camera data
  "include_depth_maps": false       // Include raw depth data
}
```

**Response (JSON format):**
```json
{
  "success": true,
  "job_id": "20241118_123456_image",
  "point_cloud": {
    "vertices": [
      [0.123, 0.456, 0.789],
      [0.234, 0.567, 0.890],
      // ... (x, y, z) coordinates
    ],
    "colors": [
      [255, 128, 64],   // RGB 0-255
      [128, 255, 192],
      // ...
    ],
    "normals": [       // Optional
      [0, 1, 0],
      [0, 1, 0],
      // ...
    ]
  },
  "cameras": [
    {
      "position": [0, 0, 5],
      "rotation": [0, 0, 0],
      "fov": 75,
      "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]
    }
  ],
  "metadata": {
    "num_points": 1000000,
    "num_cameras": 1,
    "processing_time": 2.34,
    "bounds": {
      "min": [-1.5, -1.2, -0.8],
      "max": [1.5, 1.2, 0.8]
    }
  }
}
```

**Response (GLB format):**
```
Content-Type: model/gltf-binary
Content-Disposition: attachment; filename="scene.glb"

<binary GLB data>
```

---

### 2. **POST /api/v1/process/video**
Process a video and return combined point cloud.

**Request Body:**
```json
{
  "video": "data:video/mp4;base64,AAAAIGZ0eXBpc29t...",
  // OR
  "video_url": "https://example.com/video.mp4",
  // OR
  "video_path": "/path/to/video.mp4",

  "fps": 2.0,                      // Frame extraction rate
  "resolution": 504,
  "max_points": 1000000,
  "export_format": "json",
  "include_cameras": true
}
```

**Response:** Same as image endpoint but with multiple cameras.

---

### 3. **POST /api/v1/process/batch**
Process multiple images and return combined point cloud.

**Request Body:**
```json
{
  "images": [
    {"data": "data:image/png;base64,..."},
    {"url": "https://example.com/img1.jpg"},
    {"path": "/local/img2.jpg"}
  ],
  "resolution": 504,
  "max_points": 1000000,
  "export_format": "json"
}
```

---

### 4. **GET /api/v1/job/<job_id>/status**
Get processing status for async jobs.

**Response:**
```json
{
  "job_id": "20241118_123456_video",
  "status": "processing",  // queued, processing, completed, error
  "progress": 45,          // 0-100
  "message": "Processing frame 23/50...",
  "estimated_time_remaining": 120  // seconds
}
```

---

### 5. **GET /api/v1/job/<job_id>/result**
Get result for completed job.

**Query Parameters:**
- `format`: json, glb, npz, ply

**Response:** Point cloud data in requested format.

---

### 6. **GET /api/v1/export/<job_id>/<format>**
Export point cloud in different formats.

**Formats:**
- `json` - Structured JSON with vertices/colors
- `glb` - Binary GLB file
- `ply` - PLY point cloud format
- `npz` - NumPy compressed format
- `obj` - Wavefront OBJ format

---

## 📊 Data Format Specifications

### Point Cloud JSON Format

```json
{
  "version": "1.0",
  "type": "point_cloud",
  "vertices": [[x, y, z], ...],     // Float32 arrays
  "colors": [[r, g, b], ...],       // Uint8 or Float (0-1)
  "normals": [[nx, ny, nz], ...],   // Optional
  "confidence": [0.0-1.0, ...],     // Optional confidence per point
  "metadata": {
    "coordinate_system": "opengl",  // or "opencv"
    "units": "meters",
    "num_points": 1000000,
    "bounds": {
      "min": [x, y, z],
      "max": [x, y, z],
      "center": [x, y, z]
    }
  }
}
```

### Camera JSON Format

```json
{
  "cameras": [
    {
      "id": 0,
      "type": "perspective",
      "position": [x, y, z],
      "rotation": [rx, ry, rz],     // Euler angles
      "quaternion": [x, y, z, w],    // Alternative
      "look_at": [x, y, z],
      "up": [x, y, z],
      "fov": 75,
      "aspect": 1.777,
      "near": 0.1,
      "far": 1000,
      "intrinsics": {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      },
      "extrinsics": {
        "rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
        "translation": [tx, ty, tz],
        "matrix": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]]
      }
    }
  ]
}
```

### Binary Format (Efficient)

For large point clouds, use binary format:

**Header (64 bytes):**
```
4 bytes: Magic number (0x44413350) "DA3P"
4 bytes: Version (1)
4 bytes: Num points
4 bytes: Num cameras
4 bytes: Flags (has_colors, has_normals, has_confidence)
44 bytes: Reserved
```

**Data:**
```
Vertices: num_points * 12 bytes (3 * float32)
Colors: num_points * 3 bytes (3 * uint8) [if has_colors]
Normals: num_points * 12 bytes (3 * float32) [if has_normals]
Confidence: num_points * 4 bytes (float32) [if has_confidence]
Camera data: variable
```

---

## 🔨 Implementation Steps

### Step 1: Helper Functions

```python
# Decode base64 image
def decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    if data.startswith('data:'):
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)

# Download image from URL
def download_image_from_url(url: str) -> np.ndarray:
    """Download and decode image from URL."""
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return np.array(img)

# Convert point cloud to JSON
def point_cloud_to_json(prediction: Prediction, max_points: int) -> dict:
    """Convert DA3 prediction to JSON format."""
    # Extract vertices from depth maps
    vertices, colors = depth_to_vertices(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images
    )

    # Downsample if needed
    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[indices]
        colors = colors[indices]

    return {
        "vertices": vertices.tolist(),
        "colors": colors.tolist(),
        "metadata": {
            "num_points": len(vertices),
            "bounds": {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist()
            }
        }
    }

# Extract camera data
def extract_camera_data(prediction: Prediction) -> list:
    """Extract camera parameters."""
    cameras = []
    for i in range(len(prediction.extrinsics)):
        ext = prediction.extrinsics[i]
        ixt = prediction.intrinsics[i]

        # Convert to camera position and rotation
        R = ext[:3, :3]
        t = ext[:3, 3]
        camera_pos = -R.T @ t

        cameras.append({
            "id": i,
            "position": camera_pos.tolist(),
            "intrinsics": {
                "fx": float(ixt[0, 0]),
                "fy": float(ixt[1, 1]),
                "cx": float(ixt[0, 2]),
                "cy": float(ixt[1, 2]),
                "matrix": ixt.tolist()
            },
            "extrinsics": {
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "matrix": ext.tolist()
            }
        })

    return cameras
```

### Step 2: API Route Implementations

```python
@app.route('/api/v1/process/image', methods=['POST'])
def api_process_image():
    """Process single image via API."""
    data = request.json

    # Validate request
    if not any(k in data for k in ['image', 'image_url', 'image_path']):
        return jsonify({'error': 'No image provided'}), 400

    # Load image
    if 'image' in data:
        img = decode_base64_image(data['image'])
    elif 'image_url' in data:
        img = download_image_from_url(data['image_url'])
    else:
        img_path = data['image_path']
        img = np.array(Image.open(img_path))

    # Get parameters
    resolution = data.get('resolution', 504)
    max_points = data.get('max_points', 1000000)
    export_format = data.get('export_format', 'json')

    # Process
    prediction = state.model.inference(
        image=[img],
        process_res=resolution,
        num_max_points=max_points
    )

    # Format response
    if export_format == 'json':
        result = {
            "success": True,
            "point_cloud": point_cloud_to_json(prediction, max_points),
            "cameras": extract_camera_data(prediction),
            "metadata": {
                "processing_time": time.time() - start_time,
                "num_points": len(point_cloud["vertices"])
            }
        }
        return jsonify(result)

    elif export_format == 'glb':
        # Return GLB binary
        glb_path = export_to_glb(prediction, ...)
        return send_file(glb_path, mimetype='model/gltf-binary')
```

### Step 3: Example Client Code

```python
# example_client.py
import requests
import base64
import json

# Example 1: Process image from base64
def process_image_base64(image_path):
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    response = requests.post('http://localhost:5000/api/v1/process/image', json={
        'image': f'data:image/jpeg;base64,{img_data}',
        'resolution': 504,
        'max_points': 500000,
        'export_format': 'json'
    })

    result = response.json()
    print(f"Processed {result['metadata']['num_points']} points")
    return result

# Example 2: Process image from URL
def process_image_url(url):
    response = requests.post('http://localhost:5000/api/v1/process/image', json={
        'image_url': url,
        'resolution': 756,
        'export_format': 'json'
    })
    return response.json()

# Example 3: Process video
def process_video(video_path):
    with open(video_path, 'rb') as f:
        video_data = base64.b64encode(f.read()).decode()

    response = requests.post('http://localhost:5000/api/v1/process/video', json={
        'video': f'data:video/mp4;base64,{video_data}',
        'fps': 2.0,
        'resolution': 504,
        'export_format': 'json'
    })
    return response.json()

# Example 4: Download GLB
def download_glb(image_path, output_path):
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    response = requests.post('http://localhost:5000/api/v1/process/image', json={
        'image': f'data:image/jpeg;base64,{img_data}',
        'export_format': 'glb'
    })

    with open(output_path, 'wb') as f:
        f.write(response.content)

# Example 5: Load point cloud in Three.js
def get_threejs_format(image_path):
    result = process_image_base64(image_path)

    # Convert to Three.js BufferGeometry format
    threejs_data = {
        "positions": [],  # Flat array: [x, y, z, x, y, z, ...]
        "colors": []      # Flat array: [r, g, b, r, g, b, ...]
    }

    for vertex in result['point_cloud']['vertices']:
        threejs_data['positions'].extend(vertex)

    for color in result['point_cloud']['colors']:
        # Normalize to 0-1
        threejs_data['colors'].extend([c/255.0 for c in color])

    return threejs_data
```

### Step 4: Three.js Integration Example

```javascript
// Load point cloud from API
async function loadPointCloudFromAPI(imagePath) {
    // Convert image to base64
    const imageBase64 = await fileToBase64(imagePath);

    // Send to API
    const response = await fetch('http://localhost:5000/api/v1/process/image', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: imageBase64,
            resolution: 504,
            max_points: 1000000,
            export_format: 'json'
        })
    });

    const data = await response.json();

    // Create Three.js point cloud
    const geometry = new THREE.BufferGeometry();

    // Flatten vertices array
    const positions = new Float32Array(
        data.point_cloud.vertices.flat()
    );

    // Normalize colors to 0-1
    const colors = new Float32Array(
        data.point_cloud.colors.flat().map(c => c / 255)
    );

    geometry.setAttribute('position',
        new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color',
        new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.01,
        vertexColors: true
    });

    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    return pointCloud;
}

// Or load GLB directly
async function loadGLBFromAPI(imagePath) {
    const imageBase64 = await fileToBase64(imagePath);

    const response = await fetch('http://localhost:5000/api/v1/process/image', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: imageBase64,
            export_format: 'glb'
        })
    });

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    const loader = new THREE.GLTFLoader();
    loader.load(url, (gltf) => {
        scene.add(gltf.scene);
    });
}
```

---

## 🔒 Security Considerations

1. **Rate Limiting**: Limit requests per IP
2. **File Size Limits**: Max image/video size
3. **Authentication**: Optional API key support
4. **CORS**: Configure allowed origins
5. **Input Validation**: Sanitize all inputs
6. **Resource Limits**: Max processing time, memory

---

## 📈 Performance Optimizations

1. **Async Processing**: Queue jobs for large files
2. **Caching**: Cache results by content hash
3. **Compression**: Gzip JSON responses
4. **Binary Format**: Use binary for large point clouds
5. **Streaming**: Stream large files instead of buffering

---

## 🧪 Testing

```bash
# Test image processing
curl -X POST http://localhost:5000/api/v1/process/image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg", "export_format": "json"}'

# Test with local file
python3 example_client.py --image test.jpg --format json

# Load test
ab -n 100 -c 10 -p request.json http://localhost:5000/api/v1/process/image
```

---

## 📝 Documentation

Auto-generate API docs with:
- OpenAPI/Swagger spec
- Interactive API explorer
- Code examples in multiple languages
- Postman collection

---

## Next Steps

1. Implement helper functions ✅
2. Add API routes ✅
3. Create example client ✅
4. Add error handling ✅
5. Write tests ✅
6. Document API ✅
7. Add rate limiting ⏳
8. Deploy ⏳

Ready to implement? Let's start! 🚀
