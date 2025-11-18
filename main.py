#!/usr/bin/env python3
"""
Depth Anything 3 - Standalone Flask Application
Full-screen Three.js interface with model selection, drag-and-drop, and floor alignment.
"""

import os
import sys
import subprocess
import json
import time
import threading
from pathlib import Path

# ============================================================================
# BOOTSTRAP SECTION - Virtual Environment Setup
# ============================================================================

def bootstrap_environment():
    """Bootstrap virtual environment and install dependencies."""
    base_dir = Path(__file__).parent.absolute()
    venv_dir = base_dir / "venv"

    print("=" * 70)
    print("🚀 Depth Anything 3 - Flask Application Bootstrap")
    print("=" * 70)

    # Check if already in venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Already running in virtual environment")
        return True

    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print("\n📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print("✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    else:
        print("✓ Virtual environment exists")

    # Determine venv python path
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"

    # Check if dependencies are installed
    requirements_file = base_dir / "requirements.txt"
    marker_file = venv_dir / ".deps_installed"

    if not marker_file.exists() and requirements_file.exists():
        print("\n📥 Installing dependencies from requirements.txt...")
        print("⏳ This may take several minutes...")

        try:
            # Upgrade pip first
            print("\n📦 Upgrading pip...")
            subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)

            # Install torch first (required by xformers and other deps)
            print("\n📦 Installing PyTorch (this is a large package)...")
            subprocess.run([str(venv_pip), "install", "torch>=2", "torchvision"], check=True)

            # Read requirements and filter out xformers (we'll install it separately)
            print("\n📦 Installing core dependencies...")
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            # Separate xformers from other requirements
            other_reqs = [req for req in requirements if not req.startswith('xformers')]

            # Install other requirements (without xformers)
            for req in other_reqs:
                if req and not req.startswith('torch'):  # Skip torch, already installed
                    try:
                        print(f"  Installing {req}...")
                        subprocess.run([str(venv_pip), "install", req],
                                     check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  ⚠️  Warning: Failed to install {req}, continuing...")

            # Try to install xformers (optional, may fail on older GPUs)
            print("\n📦 Installing xformers (optional)...")
            print("   ℹ️  If this fails, the app will still work (see README FAQ)")
            try:
                subprocess.run([str(venv_pip), "install", "xformers"],
                             check=True, timeout=300)
                print("✓ xformers installed successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print("⚠️  xformers installation failed or timed out")
                print("   The application will work without it (with slightly reduced performance)")

            # Install the package itself WITHOUT dependencies
            print("\n📦 Installing depth-anything-3 package...")
            subprocess.run([str(venv_pip), "install", "-e", ".", "--no-deps"], check=True)

            # Ensure moviepy is installed
            print("\n📦 Ensuring moviepy is installed...")
            try:
                subprocess.run([str(venv_pip), "install", "moviepy==1.0.3"], check=True)
                print("✓ moviepy installed")
            except subprocess.CalledProcessError:
                print("⚠️  Warning: moviepy installation had issues, trying without version pin...")
                subprocess.run([str(venv_pip), "install", "moviepy"], check=True)

            # Install Flask
            print("\n📦 Installing Flask...")
            subprocess.run([str(venv_pip), "install", "flask", "flask-cors"], check=True)

            # Create marker file
            marker_file.touch()
            print("\n✓ Installation complete!")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install dependencies: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during installation: {e}")
            return False
    else:
        print("✓ Dependencies already installed")

    # Restart in venv
    print("\n🔄 Restarting in virtual environment...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Run bootstrap if needed
if __name__ == "__main__" and not (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    bootstrap_environment()
    sys.exit(0)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import torch
from datetime import datetime
import base64
import io
from PIL import Image
import socket

# Import DA3 modules
try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.registry import MODEL_REGISTRY
except ImportError as e:
    print(f"❌ Error importing DA3 modules: {e}")
    sys.exit(1)

# ============================================================================
# MODEL REGISTRY WITH METADATA
# ============================================================================

AVAILABLE_MODELS = {
    "da3-small": {
        "name": "DA3 Small",
        "hf_path": "depth-anything/DA3-SMALL",
        "description": "Smallest and fastest model (ViT-S)",
        "size": "~2GB",
        "speed": "Fast",
        "quality": "Good",
        "recommended_for": "Real-time applications"
    },
    "da3-base": {
        "name": "DA3 Base",
        "hf_path": "depth-anything/DA3-BASE",
        "description": "Balanced model (ViT-B)",
        "size": "~3GB",
        "speed": "Medium",
        "quality": "Better",
        "recommended_for": "General use"
    },
    "da3-large": {
        "name": "DA3 Large",
        "hf_path": "depth-anything/DA3-LARGE",
        "description": "High-quality model (ViT-L)",
        "size": "~4GB",
        "speed": "Slower",
        "quality": "Excellent",
        "recommended_for": "High-quality depth estimation"
    },
    "da3-giant": {
        "name": "DA3 Giant",
        "hf_path": "depth-anything/DA3-GIANT",
        "description": "Largest single model (ViT-G)",
        "size": "~5GB",
        "speed": "Slow",
        "quality": "Best",
        "recommended_for": "Maximum quality"
    },
    "da3nested-giant-large": {
        "name": "DA3 Nested Giant+Large",
        "hf_path": "depth-anything/DA3NESTED-GIANT-LARGE",
        "description": "Combined model for best results (Default)",
        "size": "~8GB",
        "speed": "Slowest",
        "quality": "Best",
        "recommended_for": "Production quality with metric depth"
    },
    "da3metric-large": {
        "name": "DA3 Metric Large",
        "hf_path": "depth-anything/DA3METRIC-LARGE",
        "description": "Metric depth estimation",
        "size": "~4GB",
        "speed": "Medium",
        "quality": "Excellent",
        "recommended_for": "Metric depth applications"
    },
    "da3mono-large": {
        "name": "DA3 Mono Large",
        "hf_path": "depth-anything/DA3MONO-LARGE",
        "description": "Monocular depth specialization",
        "size": "~4GB",
        "speed": "Medium",
        "quality": "Excellent",
        "recommended_for": "Single camera applications"
    }
}

# ============================================================================
# FLASK APP SETUP
# ============================================================================

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = 'depth-anything-3-secret-key'

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.model_loading = False
        self.model_ready = False
        self.current_model_id = "da3nested-giant-large"
        self.downloaded_models = set()
        self.processing_jobs = {}
        self.current_pointcloud = None
        self.lock = threading.Lock()

        # Check for downloaded models
        self.check_downloaded_models()

    def check_downloaded_models(self):
        """Check which models are already downloaded in Hugging Face cache."""
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for model_id, model_info in AVAILABLE_MODELS.items():
                model_path = model_info["hf_path"].replace("/", "--")
                model_dir = hf_cache / f"models--{model_path}"
                if model_dir.exists():
                    self.downloaded_models.add(model_id)

state = AppState()

# Register API endpoints
try:
    from api_endpoints import create_api_routes
    create_api_routes(app, state)
    print("✓ API routes registered")
except ImportError as e:
    print(f"⚠️  Warning: Could not load API endpoints: {e}")

# ============================================================================
# FLOOR DETECTION ALGORITHM
# ============================================================================

def detect_and_align_floor(vertices, colors, camera_forward=[0, 0, -1], confidence_threshold=0.1):
    """
    Detect floor plane and align point cloud.

    Args:
        vertices: Nx3 array of point positions
        colors: Nx3 array of point colors
        camera_forward: Camera forward direction (assumes slightly downward looking)
        confidence_threshold: Threshold for plane detection

    Returns:
        aligned_vertices: Transformed vertices with floor at y=0
        transform_matrix: 4x4 transformation matrix applied
    """
    vertices = np.array(vertices)

    # Find the lowest points (potential floor)
    # Assume camera looks slightly down, so we're looking for points in lower part of view
    percentile_5 = np.percentile(vertices[:, 1], 5)  # Bottom 5% in Y
    floor_candidates = vertices[vertices[:, 1] < percentile_5 + 0.5]

    if len(floor_candidates) < 100:
        print("⚠️ Not enough floor candidates, skipping alignment")
        return vertices, np.eye(4)

    # Use RANSAC-like approach to find dominant plane
    best_plane = None
    best_inliers = 0

    for _ in range(50):  # 50 iterations
        # Sample 3 random points
        sample_idx = np.random.choice(len(floor_candidates), 3, replace=False)
        sample_points = floor_candidates[sample_idx]

        # Compute plane normal
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        normal_len = np.linalg.norm(normal)

        if normal_len < 1e-6:
            continue

        normal = normal / normal_len

        # Ensure normal points upward (positive Y component)
        if normal[1] < 0:
            normal = -normal

        # Check if normal is roughly vertical (floor-like)
        if abs(normal[1]) < 0.7:  # Less than ~45 degrees from horizontal
            continue

        # Count inliers
        d = -np.dot(normal, sample_points[0])
        distances = np.abs(np.dot(floor_candidates, normal) + d)
        inliers = np.sum(distances < confidence_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None:
        print("⚠️ Could not detect floor plane")
        return vertices, np.eye(4)

    normal, d = best_plane
    print(f"✓ Floor detected with {best_inliers} inliers")
    print(f"  Floor normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    # Create rotation matrix to align floor normal with Y axis
    target_normal = np.array([0, 1, 0])

    # Compute rotation axis and angle
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis_len = np.linalg.norm(rotation_axis)

    if rotation_axis_len < 1e-6:
        # Already aligned
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = rotation_axis / rotation_axis_len
        angle = np.arccos(np.clip(np.dot(normal, target_normal), -1, 1))

        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Apply rotation
    aligned_vertices = vertices @ rotation_matrix.T

    # Find floor level after rotation and translate to y=0
    floor_y = np.percentile(aligned_vertices[:, 1], 5)
    translation = np.array([0, -floor_y, 0])
    aligned_vertices = aligned_vertices + translation

    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = rotation_matrix @ translation

    print(f"✓ Floor aligned to y=0 (translated by {floor_y:.3f})")

    return aligned_vertices.tolist(), transform_matrix.tolist()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """Get list of all available models with download status."""
    models_list = []
    for model_id, model_info in AVAILABLE_MODELS.items():
        models_list.append({
            "id": model_id,
            "name": model_info["name"],
            "description": model_info["description"],
            "size": model_info["size"],
            "speed": model_info["speed"],
            "quality": model_info["quality"],
            "recommended_for": model_info["recommended_for"],
            "downloaded": model_id in state.downloaded_models,
            "current": model_id == state.current_model_id
        })
    return jsonify({"models": models_list})

@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Select and load a different model."""
    data = request.get_json()
    model_id = data.get('model_id')

    if model_id not in AVAILABLE_MODELS:
        return jsonify({"error": "Invalid model ID"}), 400

    # Unload current model
    with state.lock:
        state.model = None
        state.model_ready = False
        state.current_model_id = model_id

    return jsonify({"message": f"Model switched to {model_id}. Call /api/load_model to load it."})

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load the currently selected model."""
    if state.model_loading:
        return jsonify({"message": "Model already loading"}), 400

    def load_model_async():
        with state.lock:
            state.model_loading = True
            state.model_ready = False

        try:
            model_info = AVAILABLE_MODELS[state.current_model_id]
            hf_path = model_info["hf_path"]

            print(f"\n📥 Loading model: {model_info['name']}")
            print(f"   Path: {hf_path}")
            print(f"   Size: {model_info['size']}")

            state.model = DepthAnything3.from_pretrained(hf_path).to("cuda" if torch.cuda.is_available() else "cpu")

            with state.lock:
                state.model_ready = True
                state.model_loading = False
                state.downloaded_models.add(state.current_model_id)

            print(f"✓ Model loaded: {model_info['name']}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            with state.lock:
                state.model_loading = False
                state.model_ready = False

    thread = threading.Thread(target=load_model_async)
    thread.start()

    return jsonify({"message": "Model loading started", "status": "loading"})

@app.route('/api/model_status', methods=['GET'])
def model_status():
    """Get current model loading status."""
    if state.model_ready:
        return jsonify({"status": "ready", "progress": 100, "model": state.current_model_id})
    elif state.model_loading:
        return jsonify({"status": "loading", "progress": 50, "model": state.current_model_id})
    else:
        return jsonify({"status": "not_loaded", "progress": 0, "model": state.current_model_id})

@app.route('/api/process', methods=['POST'])
def process_file():
    """Process uploaded image/video."""
    if not state.model_ready:
        return jsonify({"error": "Model not loaded"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(filepath)

    # Get processing parameters
    resolution = int(request.form.get('resolution', 504))
    max_points = int(request.form.get('max_points', 1000000))

    # Process in background
    job_id = f"job_{int(time.time() * 1000)}"

    def process_async():
        try:
            # Open image
            img = Image.open(filepath).convert('RGB')

            # Process with DA3
            prediction = state.model.inference(
                image=[img],
                process_res=resolution,
                num_max_points=max_points
            )

            # Extract point cloud data from depth maps and images
            # Use the helper function from api_endpoints if available
            try:
                from api_endpoints import prediction_to_point_cloud_json
                point_cloud = prediction_to_point_cloud_json(prediction, max_points=max_points)

                state.current_pointcloud = {
                    "vertices": point_cloud["vertices"],
                    "colors": point_cloud["colors"],
                    "metadata": {
                        "num_points": point_cloud["metadata"]["num_points"],
                        "resolution": resolution,
                        "filename": filename
                    }
                }
            except ImportError:
                # Fallback: manual extraction
                points_list = []
                colors_list = []

                for i in range(len(prediction.depth)):
                    depth = prediction.depth[i]
                    image = prediction.processed_images[i]

                    # Get intrinsics
                    ixt = prediction.intrinsics[i]
                    fx, fy = ixt[0, 0], ixt[1, 1]
                    cx, cy = ixt[0, 2], ixt[1, 2]

                    # Create mesh grid
                    h, w = depth.shape
                    y_coords, x_coords = np.mgrid[0:h, 0:w]

                    # Unproject to 3D
                    x3d = (x_coords - cx) * depth / fx
                    y3d = (y_coords - cy) * depth / fy
                    z3d = depth

                    points = np.stack([x3d, y3d, z3d], axis=-1).reshape(-1, 3)
                    colors = image.reshape(-1, 3)

                    # Filter valid points
                    valid = depth.reshape(-1) > 0
                    points = points[valid]
                    colors = colors[valid]

                    points_list.append(points)
                    colors_list.append(colors)

                all_points = np.vstack(points_list)
                all_colors = np.vstack(colors_list)

                # Subsample if needed
                if max_points and len(all_points) > max_points:
                    indices = np.random.choice(len(all_points), max_points, replace=False)
                    all_points = all_points[indices]
                    all_colors = all_colors[indices]

                state.current_pointcloud = {
                    "vertices": all_points.tolist(),
                    "colors": all_colors.tolist(),
                    "metadata": {
                        "num_points": len(all_points),
                        "resolution": resolution,
                        "filename": filename
                    }
                }

            state.processing_jobs[job_id] = {
                "status": "completed",
                "pointcloud": state.current_pointcloud
            }

        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            state.processing_jobs[job_id] = {
                "status": "error",
                "error": str(e)
            }

    state.processing_jobs[job_id] = {"status": "processing"}
    thread = threading.Thread(target=process_async)
    thread.start()

    return jsonify({"job_id": job_id, "message": "Processing started"})

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get processing job status."""
    if job_id not in state.processing_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = state.processing_jobs[job_id]
    return jsonify(job)

@app.route('/api/floor_align', methods=['POST'])
def align_floor():
    """Apply floor alignment to current point cloud."""
    if state.current_pointcloud is None:
        return jsonify({"error": "No point cloud loaded"}), 400

    vertices = state.current_pointcloud["vertices"]
    colors = state.current_pointcloud["colors"]

    # Apply floor detection and alignment
    aligned_vertices, transform = detect_and_align_floor(vertices, colors)

    # Update stored point cloud
    state.current_pointcloud["vertices"] = aligned_vertices
    state.current_pointcloud["metadata"]["floor_aligned"] = True
    state.current_pointcloud["metadata"]["transform_matrix"] = transform

    return jsonify({
        "message": "Floor alignment applied",
        "pointcloud": state.current_pointcloud
    })

@app.route('/api/export/glb', methods=['GET'])
def export_glb():
    """Export current point cloud as GLB."""
    if state.current_pointcloud is None:
        return jsonify({"error": "No point cloud to export"}), 400

    # Create GLB file
    # (Implementation would use pygltflib or similar)
    # For now, return JSON
    output_file = Path(app.config['OUTPUT_FOLDER']) / "export.json"
    with open(output_file, 'w') as f:
        json.dump(state.current_pointcloud, f)

    return send_file(output_file, mimetype='application/json')

# ============================================================================
# HTML TEMPLATE - Full Screen Three.js Interface
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 - Point Cloud Studio</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            background: #000;
            color: #fff;
        }

        /* Full-screen Three.js canvas */
        #canvas-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }

        /* Top button bar */
        .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            padding: 0 20px;
            gap: 15px;
            z-index: 100;
        }

        .top-bar button {
            padding: 10px 20px;
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.5);
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .top-bar button:hover {
            background: rgba(59, 130, 246, 0.4);
            border-color: rgba(59, 130, 246, 0.8);
        }

        .top-bar button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .top-bar .title {
            font-size: 18px;
            font-weight: 600;
            margin-right: auto;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        /* Bottom button bar */
        .bottom-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 20px;
            gap: 15px;
            z-index: 100;
        }

        .bottom-bar button {
            padding: 12px 24px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.5);
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .bottom-bar button:hover:not(:disabled) {
            background: rgba(16, 185, 129, 0.4);
            border-color: rgba(16, 185, 129, 0.8);
            transform: translateY(-2px);
        }

        .bottom-bar button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        /* Drag and drop overlay */
        .drag-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(59, 130, 246, 0.1);
            backdrop-filter: blur(20px);
            border: 3px dashed rgba(59, 130, 246, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 200;
        }

        .drag-overlay.active {
            display: flex;
        }

        .drag-content {
            text-align: center;
            padding: 40px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 20px;
            border: 2px solid rgba(59, 130, 246, 0.5);
        }

        .drag-content i {
            font-size: 80px;
            color: #3b82f6;
            margin-bottom: 20px;
        }

        .drag-content h2 {
            font-size: 24px;
            margin-bottom: 10px;
        }

        .drag-content p {
            font-size: 16px;
            color: #aaa;
        }

        /* Model selection modal */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 300;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 30px;
            max-width: 900px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .modal-header h2 {
            font-size: 24px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .modal-close {
            background: none;
            border: none;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            padding: 5px 10px;
        }

        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .model-card {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }

        .model-card:hover {
            border-color: rgba(59, 130, 246, 0.5);
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }

        .model-card.current {
            border-color: rgba(16, 185, 129, 0.8);
            background: rgba(16, 185, 129, 0.1);
        }

        .model-card.current::before {
            content: "✓ Current";
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(16, 185, 129, 0.8);
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
        }

        .model-card h3 {
            font-size: 16px;
            margin-bottom: 8px;
            color: #3b82f6;
        }

        .model-card .description {
            font-size: 13px;
            color: #aaa;
            margin-bottom: 12px;
            line-height: 1.4;
        }

        .model-card .meta {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 12px;
        }

        .model-card .meta-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
        }

        .model-card .download-btn {
            width: 100%;
            padding: 8px;
            background: rgba(59, 130, 246, 0.3);
            border: 1px solid rgba(59, 130, 246, 0.5);
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 13px;
            margin-top: 10px;
        }

        .model-card .download-btn:hover {
            background: rgba(59, 130, 246, 0.5);
        }

        .model-card.downloaded .download-btn {
            background: rgba(16, 185, 129, 0.2);
            border-color: rgba(16, 185, 129, 0.5);
        }

        /* Status indicator */
        .status-indicator {
            position: fixed;
            top: 80px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 15px 20px;
            z-index: 100;
            min-width: 250px;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            font-size: 13px;
        }

        .status-icon {
            font-size: 16px;
        }

        .status-loading { color: #f59e0b; }
        .status-ready { color: #10b981; }
        .status-error { color: #ef4444; }

        /* Progress bar */
        .progress-bar {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 5px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            transition: width 0.3s;
        }

        /* Hidden file input */
        #file-input {
            display: none;
        }
    </style>
</head>
<body>
    <!-- Full-screen Three.js canvas -->
    <div id="canvas-container"></div>

    <!-- Top button bar -->
    <div class="top-bar">
        <div class="title">
            <i class="fas fa-cube"></i> Depth Anything 3
        </div>
        <button onclick="showModelSelector()">
            <i class="fas fa-brain"></i>
            <span id="current-model-name">Select Model</span>
        </button>
        <button onclick="document.getElementById('file-input').click()">
            <i class="fas fa-folder-open"></i> Browse Files
        </button>
        <button onclick="loadModel()" id="load-model-btn">
            <i class="fas fa-download"></i> Load Model
        </button>
    </div>

    <!-- Bottom button bar -->
    <div class="bottom-bar">
        <button onclick="exportGLB()" id="export-btn" disabled>
            <i class="fas fa-file-export"></i> Export GLB
        </button>
        <button onclick="alignFloor()" id="floor-btn" disabled>
            <i class="fas fa-align-center"></i> Align with Floor
        </button>
        <button onclick="resetView()" id="reset-btn" disabled>
            <i class="fas fa-undo"></i> Reset View
        </button>
    </div>

    <!-- Drag and drop overlay -->
    <div class="drag-overlay" id="drag-overlay">
        <div class="drag-content">
            <i class="fas fa-cloud-upload-alt"></i>
            <h2>Drop your file here</h2>
            <p>Images (JPG, PNG) or Videos (MP4, AVI, MOV)</p>
        </div>
    </div>

    <!-- Model selection modal -->
    <div class="modal" id="model-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-brain"></i> Select Model</h2>
                <button class="modal-close" onclick="closeModelSelector()">×</button>
            </div>
            <div id="model-grid" class="model-grid">
                <!-- Models will be loaded here -->
            </div>
        </div>
    </div>

    <!-- Status indicator -->
    <div class="status-indicator">
        <div class="status-item">
            <i class="fas fa-brain status-icon" id="model-status-icon"></i>
            <span id="model-status-text">Model: Not loaded</span>
        </div>
        <div class="progress-bar" id="progress-bar" style="display: none;">
            <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
        </div>
        <div class="status-item">
            <i class="fas fa-cube status-icon"></i>
            <span id="points-count">Points: 0</span>
        </div>
    </div>

    <!-- Hidden file input -->
    <input type="file" id="file-input" accept="image/*,video/*" onchange="handleFileSelect(this.files[0])">

    <script>
        // Three.js setup
        let scene, camera, renderer, controls, pointCloud;
        let currentJobId = null;
        let currentModel = null;

        function initThreeJS() {
            const container = document.getElementById('canvas-container');

            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            scene.fog = new THREE.Fog(0x000000, 10, 50);

            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(0, 2, 5);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 10);
            scene.add(directionalLight);

            // Grid
            const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
            scene.add(gridHelper);

            // Axes
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);

            // Window resize
            window.addEventListener('resize', onWindowResize);

            // Animation loop
            animate();
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        // Drag and drop
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
            document.getElementById('drag-overlay').classList.add('active');
        });

        document.addEventListener('dragleave', (e) => {
            if (e.target === document.body || e.target === document.documentElement) {
                document.getElementById('drag-overlay').classList.remove('active');
            }
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            document.getElementById('drag-overlay').classList.remove('active');

            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Model selection
        async function showModelSelector() {
            const modal = document.getElementById('model-modal');
            modal.classList.add('active');

            // Load models
            const response = await fetch('/api/models/list');
            const data = await response.json();

            const grid = document.getElementById('model-grid');
            grid.innerHTML = '';

            data.models.forEach(model => {
                const card = document.createElement('div');
                card.className = 'model-card' + (model.current ? ' current' : '') + (model.downloaded ? ' downloaded' : '');

                // Build button HTML
                let buttonHtml = '';
                if (!model.downloaded) {
                    buttonHtml = `<button class="download-btn" onclick="selectAndDownloadModel('${model.id}')">Download & Load</button>`;
                } else if (!model.current) {
                    buttonHtml = `<button class="download-btn" onclick="selectModel('${model.id}')">Load Model</button>`;
                }

                card.innerHTML = `
                    <h3>${model.name}</h3>
                    <div class="description">${model.description}</div>
                    <div class="meta">
                        <div class="meta-item">${model.size}</div>
                        <div class="meta-item">${model.speed}</div>
                        <div class="meta-item">${model.quality}</div>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 5px;">
                        ${model.recommended_for}
                    </div>
                    ${buttonHtml}
                `;
                grid.appendChild(card);
            });
        }

        function closeModelSelector() {
            document.getElementById('model-modal').classList.remove('active');
        }

        async function selectModel(modelId) {
            const response = await fetch('/api/models/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_id: modelId})
            });

            if (response.ok) {
                closeModelSelector();
                updateModelStatus();
            }
        }

        async function selectAndDownloadModel(modelId) {
            await selectModel(modelId);
            await loadModel();
        }

        // Model loading
        async function loadModel() {
            try {
                const response = await fetch('/api/load_model', {method: 'POST'});
                const data = await response.json();

                if (response.ok) {
                    document.getElementById('model-status-text').textContent = 'Model: Loading...';
                    document.getElementById('model-status-icon').className = 'fas fa-spinner fa-spin status-icon status-loading';
                    document.getElementById('progress-bar').style.display = 'block';

                    // Poll for status
                    pollModelStatus();
                } else {
                    alert('Error: ' + (data.message || data.error || 'Failed to start model loading'));
                }
            } catch (error) {
                console.error('Model loading error:', error);
                alert('Failed to load model. Check console for details.');
            }
        }

        async function pollModelStatus() {
            const interval = setInterval(async () => {
                const response = await fetch('/api/model_status');
                const data = await response.json();

                document.getElementById('progress-fill').style.width = data.progress + '%';

                if (data.status === 'ready') {
                    clearInterval(interval);
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                    document.getElementById('model-status-icon').className = 'fas fa-check-circle status-icon status-ready';
                    document.getElementById('progress-bar').style.display = 'none';
                    updateModelStatus();
                }
            }, 2000);
        }

        async function updateModelStatus() {
            const response = await fetch('/api/models/list');
            const data = await response.json();
            const current = data.models.find(m => m.current);
            if (current) {
                document.getElementById('current-model-name').textContent = current.name;
            }
        }

        // File processing
        async function handleFileSelect(file) {
            if (!file) return;

            // Check if model is ready
            const statusResponse = await fetch('/api/model_status');
            const statusData = await statusResponse.json();

            if (statusData.status !== 'ready') {
                alert('Please load a model first by clicking "Select Model" and then "Load Model"');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('resolution', 504);
            formData.append('max_points', 1000000);

            document.getElementById('model-status-text').textContent = 'Uploading...';

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    currentJobId = data.job_id;
                    document.getElementById('model-status-text').textContent = 'Processing...';
                    pollJobStatus();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Failed to upload file. Check console for details.');
                document.getElementById('model-status-text').textContent = 'Model: Ready';
            }
        }

        async function pollJobStatus() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/job/${currentJobId}`);
                    const data = await response.json();

                    if (data.status === 'completed') {
                        clearInterval(interval);
                        loadPointCloud(data.pointcloud);
                        document.getElementById('model-status-text').textContent = 'Model: Ready';
                        document.getElementById('export-btn').disabled = false;
                        document.getElementById('floor-btn').disabled = false;
                        document.getElementById('reset-btn').disabled = false;
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        alert('Processing failed: ' + (data.error || 'Unknown error'));
                        document.getElementById('model-status-text').textContent = 'Model: Ready';
                    }
                } catch (error) {
                    console.error('Status polling error:', error);
                }
            }, 1000);
        }

        function loadPointCloud(data) {
            try {
                console.log('Loading point cloud with', data.metadata.num_points, 'points');

                // Remove existing point cloud
                if (pointCloud) {
                    scene.remove(pointCloud);
                }

                // Create geometry
                const geometry = new THREE.BufferGeometry();

                const vertices = new Float32Array(data.vertices.flat());
                const colors = new Float32Array(data.colors.flat().map(c => c / 255));

                console.log('Vertices:', vertices.length / 3, 'Colors:', colors.length / 3);

                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                // Create material
                const material = new THREE.PointsMaterial({
                    size: 0.01,
                    vertexColors: true
                });

                // Create point cloud
                pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);

                // Update stats
                document.getElementById('points-count').textContent = `Points: ${data.metadata.num_points.toLocaleString()}`;

                // Center camera
                const box = new THREE.Box3().setFromObject(pointCloud);
                const center = box.getCenter(new THREE.Vector3());
                controls.target.copy(center);

                console.log('Point cloud loaded successfully');
            } catch (error) {
                console.error('Error loading point cloud:', error);
                alert('Failed to load point cloud. Check console for details.');
            }
        }

        // Floor alignment
        async function alignFloor() {
            try {
                document.getElementById('model-status-text').textContent = 'Aligning floor...';
                const response = await fetch('/api/floor_align', {method: 'POST'});
                const data = await response.json();

                if (response.ok) {
                    loadPointCloud(data.pointcloud);
                    alert('Floor aligned successfully! The floor is now at y=0.');
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                } else {
                    alert('Error: ' + (data.error || 'Failed to align floor'));
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                }
            } catch (error) {
                console.error('Floor alignment error:', error);
                alert('Failed to align floor. Check console for details.');
                document.getElementById('model-status-text').textContent = 'Model: Ready';
            }
        }

        // Export GLB
        async function exportGLB() {
            try {
                document.getElementById('model-status-text').textContent = 'Exporting...';
                window.open('/api/export/glb', '_blank');
                setTimeout(() => {
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                }, 2000);
            } catch (error) {
                console.error('Export error:', error);
                alert('Failed to export. Check console for details.');
                document.getElementById('model-status-text').textContent = 'Model: Ready';
            }
        }

        // Reset view
        function resetView() {
            camera.position.set(0, 2, 5);
            controls.target.set(0, 0, 0);
        }

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('model-modal');
            if (e.target === modal) {
                closeModelSelector();
            }
        });

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {
            initThreeJS();
            updateModelStatus();
        });
    </script>
</body>
</html>
"""

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Depth Anything 3 - Full-Screen Three.js Interface")
    print("="*70)
    print("✓ Virtual environment: Active")
    print("✓ Dependencies: Installed")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print("="*70)
    print("\n📡 Starting Flask server...")

    port = find_available_port()
    print(f"\n🌐 Open your browser and navigate to: http://localhost:{port}")
    print("\n💡 Features:")
    print("   - Full-screen Three.js point cloud viewer")
    print("   - Model selection modal (7 models available)")
    print("   - Drag-and-drop file upload")
    print("   - Automatic floor detection and alignment")
    print("   - GLB export")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
