# 🌐 Depth Anything 3 - RESTful API Guide

## Overview

The Depth Anything 3 Flask application now includes a comprehensive RESTful API for programmatic access to point cloud generation from images and videos.

---

## 🚀 Quick Start

### 1. Start the Server

```bash
python3 main.py
```

The server will start on port 5000 (or next available port) and automatically register the API endpoints.

### 2. Check Health

```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "depth-anything/DA3NESTED-GIANT-LARGE",
  "cuda_available": true
}
```

### 3. Process Your First Image

```python
import requests
import base64

# Encode image
with open('test.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:5000/api/v1/process/image', json={
    'image': f'data:image/jpeg;base64,{img_b64}',
    'resolution': 504,
    'export_format': 'json'
})

result = response.json()
print(f"Points: {result['point_cloud']['metadata']['num_points']}")
```

---

## 📡 API Endpoints

### **POST /api/v1/process/image**

Process a single image and return point cloud data.

**Request Body:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  // OR "image_url": "https://example.com/image.jpg",
  // OR "image_path": "/path/to/image.jpg",

  "resolution": 504,              // Optional: 504, 756, 1024
  "max_points": 1000000,          // Optional: max points in cloud
  "export_format": "json",        // Optional: "json" or "glb"
  "include_cameras": true,        // Optional: include camera data
  "include_confidence": false     // Optional: include confidence values
}
```

**Response (JSON format):**

```json
{
  "success": true,
  "point_cloud": {
    "vertices": [[x, y, z], ...],
    "colors": [[r, g, b], ...],
    "metadata": {
      "num_points": 1000000,
      "bounds": {
        "min": [-1.5, -1.2, -0.8],
        "max": [1.5, 1.2, 0.8],
        "center": [0, 0, 0]
      }
    }
  },
  "cameras": [
    {
      "id": 0,
      "position": [x, y, z],
      "intrinsics": {...},
      "extrinsics": {...}
    }
  ],
  "metadata": {
    "processing_time": 2.34,
    "resolution": 504,
    "max_points": 1000000
  }
}
```

**Response (GLB format):**
- Binary GLB file download
- Content-Type: `model/gltf-binary`

---

### **POST /api/v1/process/video**

Process a video and return combined point cloud.

**Request Body:**

```json
{
  "video": "data:video/mp4;base64,AAAA...",
  // OR "video_url": "https://example.com/video.mp4",
  // OR "video_path": "/path/to/video.mp4",

  "fps": 2.0,                     // Frame extraction rate
  "resolution": 504,
  "max_points": 1000000,
  "export_format": "json"
}
```

**Response:** Same structure as image endpoint, with multiple cameras.

---

### **POST /api/v1/process/batch**

Process multiple images in batch.

**Request Body:**

```json
{
  "images": [
    {"data": "data:image/jpeg;base64,..."},
    {"url": "https://example.com/img.jpg"},
    {"path": "/local/path/img.jpg"}
  ],
  "resolution": 504,
  "max_points": 1000000
}
```

---

### **GET /api/v1/health**

Health check and model status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "depth-anything/DA3NESTED-GIANT-LARGE",
  "cuda_available": true
}
```

---

## 📝 Examples

### Python Client

See [example_api_client.py](example_api_client.py) for comprehensive examples:

```bash
# Process image
python3 example_api_client.py --example 1 --image test.jpg

# Download GLB
python3 example_api_client.py --example 3 --image test.jpg

# Process video
python3 example_api_client.py --example 4 --video test.mp4

# Run all examples
python3 example_api_client.py --all --image test.jpg
```

### Three.js Client

Open [example_threejs_client.html](example_threejs_client.html) in a browser:

```bash
# Start server
python3 main.py

# Open HTML file in browser
firefox example_threejs_client.html
# or
google-chrome example_threejs_client.html
```

### Curl Examples

```bash
# Test all endpoints
./test_api.sh

# Test with specific image
./test_api.sh path/to/image.jpg
```

---

## 🎨 Three.js Integration

### Load Point Cloud from API

```javascript
async function loadPointCloud(imagePath) {
    // Convert to base64
    const imageBase64 = await fileToBase64(imagePath);

    // Request from API
    const response = await fetch('http://localhost:5000/api/v1/process/image', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: imageBase64,
            resolution: 504,
            export_format': 'json'
        })
    });

    const data = await response.json();

    // Create Three.js geometry
    const geometry = new THREE.BufferGeometry();

    const positions = new Float32Array(data.point_cloud.vertices.flat());
    const colors = new Float32Array(
        data.point_cloud.colors.flat().map(c => c / 255)
    );

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.01,
        vertexColors: true
    });

    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
}
```

### Load GLB Directly

```javascript
async function loadGLB(imagePath) {
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

## 🔧 Advanced Usage

### Custom Processing Parameters

```python
response = requests.post('http://localhost:5000/api/v1/process/image', json={
    'image': image_base64,
    'resolution': 1024,          # Higher quality
    'max_points': 2000000,       # More points
    'include_cameras': True,
    'include_confidence': True,   # Get confidence values
    'export_format': 'json'
})

# Access confidence data
confidence = result['point_cloud']['confidence']
```

### Processing Multiple Images

```python
images = [
    {'data': encode_image('img1.jpg')},
    {'url': 'https://example.com/img2.jpg'},
    {'path': '/local/img3.jpg'}
]

response = requests.post('http://localhost:5000/api/v1/process/batch', json={
    'images': images,
    'resolution': 756,
    'max_points': 1500000
})
```

### Streaming Video Processing

For large videos, process in chunks:

```python
from moviepy.editor import VideoFileClip

def process_video_chunked(video_path, chunk_size=10):
    """Process video in chunks to avoid memory issues."""
    clip = VideoFileClip(video_path)
    duration = clip.duration

    results = []
    for start in range(0, int(duration), chunk_size):
        end = min(start + chunk_size, duration)
        subclip = clip.subclip(start, end)

        # Extract frames
        frames = [subclip.get_frame(t) for t in range(0, chunk_size, 1)]

        # Process chunk
        response = requests.post('http://localhost:5000/api/v1/process/batch', json={
            'images': [{'data': frame_to_base64(f)} for f in frames],
            'resolution': 504
        })

        results.append(response.json())

    return results
```

---

## 📊 Data Format Reference

### Point Cloud Format

```json
{
  "vertices": [[x, y, z], ...],      // 3D coordinates
  "colors": [[r, g, b], ...],        // RGB 0-255
  "confidence": [0.0-1.0, ...],      // Optional
  "metadata": {
    "num_points": 1000000,
    "coordinate_system": "opengl",
    "units": "arbitrary",
    "bounds": {
      "min": [x, y, z],
      "max": [x, y, z],
      "center": [x, y, z]
    }
  }
}
```

### Camera Format

```json
{
  "id": 0,
  "type": "perspective",
  "position": [x, y, z],
  "intrinsics": {
    "fx": 500.0,
    "fy": 500.0,
    "cx": 320.0,
    "cy": 240.0,
    "matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "extrinsics": {
    "rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "translation": [tx, ty, tz]
  }
}
```

---

## ⚠️ Error Handling

### Common Errors

**503 Service Unavailable**
```json
{"error": "Model not loaded. Call /api/load_model first."}
```
Solution: Load the model first via the web UI or `/api/load_model` endpoint.

**400 Bad Request**
```json
{"error": "No image provided. Use 'image' (base64), 'image_url', or 'image_path'"}
```
Solution: Include one of the image input methods.

**500 Internal Server Error**
```json
{"error": "Processing failed: <details>"}
```
Solution: Check image format, size, and server logs.

### Retry Logic

```python
import time

def process_with_retry(image_path, max_retries=3):
    for i in range(max_retries):
        try:
            response = requests.post(...)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)  # Exponential backoff
```

---

## 🚀 Performance Tips

1. **Use appropriate resolution:**
   - 504px: Fast, good for previews
   - 756px: Balanced quality/speed
   - 1024px: High quality, slower

2. **Limit point cloud size:**
   - Use `max_points` to control output size
   - Fewer points = faster processing and smaller responses

3. **Use GLB for large clouds:**
   - Binary format is more efficient than JSON
   - Better for downloading/storing

4. **Batch processing:**
   - Process multiple images in one request
   - Reduces overhead

5. **Keep model loaded:**
   - First request loads model (~5GB)
   - Subsequent requests are much faster

---

## 📚 API Reference

| Endpoint | Method | Input | Output | Purpose |
|----------|--------|-------|--------|---------|
| `/api/v1/health` | GET | - | Status JSON | Check if API is ready |
| `/api/v1/process/image` | POST | Image data | Point cloud | Process single image |
| `/api/v1/process/video` | POST | Video data | Point cloud | Process video frames |
| `/api/v1/process/batch` | POST | Image array | Point cloud | Process multiple images |
| `/api/load_model` | POST | - | Status | Load DA3 model |
| `/api/model_status` | GET | - | Status | Check model status |

---

## 🔒 Security Notes

1. **CORS is enabled** - Configure for production
2. **No authentication** - Add auth for production use
3. **File size limits** - 500MB max upload
4. **Rate limiting** - Not implemented (add for production)

---

## 📖 Additional Resources

- [Full Implementation Plan](API_IMPLEMENTATION_PLAN.md)
- [Example Python Client](example_api_client.py)
- [Example Three.js Client](example_threejs_client.html)
- [Test Script](test_api.sh)
- [Main Documentation](README.md)

---

## 🆘 Support

For issues and questions:
- Check server logs for errors
- Ensure model is loaded (`/api/v1/health`)
- Verify image format and size
- See [troubleshooting guide](QUICKSTART.md#troubleshooting)

Enjoy building with the Depth Anything 3 API! 🚀
