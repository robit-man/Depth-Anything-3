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
import trimesh

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
        self.model_name = AVAILABLE_MODELS.get(self.current_model_id, {}).get("name", self.current_model_id)
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

def _select_model_by_id(model_id):
    if model_id not in AVAILABLE_MODELS:
        return False, jsonify({"error": "Invalid model ID"}), 400

    with state.lock:
        state.model = None
        state.model_ready = False
        state.current_model_id = model_id
        state.model_name = AVAILABLE_MODELS[model_id]["name"]
    return True, None, None

@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Select and load a different model."""
    data = request.get_json()
    model_id = (data or {}).get('model_id')

    ok, resp, code = _select_model_by_id(model_id)
    if not ok:
        return resp, code

    return jsonify({"message": f"Model switched to {model_id}. Call /api/load_model to load it.", "model_id": model_id})

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load the currently selected model."""
    if state.model_loading:
        return jsonify({"message": "Model already loading"}), 400

        # No model selected
    if state.current_model_id not in AVAILABLE_MODELS:
        return jsonify({"error": "No valid model selected"}), 400

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
    """Process uploaded image/video/GLB."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(filepath)

    # If GLB/GLTF/pointcloud JSON, bypass inference and just load the point cloud
    suffix = filepath.suffix.lower()
    if suffix in {".glb", ".gltf", ".json"}:
        try:
            if suffix == ".json":
                with open(filepath, "r") as f:
                    data = json.load(f)
                if "vertices" in data and "colors" in data:
                    vertices = np.array(data["vertices"], dtype=np.float32)
                    colors = np.array(data["colors"], dtype=np.float32)
                else:
                    return jsonify({"error": "JSON missing vertices/colors"}), 400
            else:
                mesh = trimesh.load(str(filepath))
                vertices = []
                colors = []

                if isinstance(mesh, trimesh.Scene):
                    for geom in mesh.geometry.values():
                        if isinstance(geom, trimesh.points.PointCloud):
                            vertices.append(geom.vertices)
                            if geom.colors is not None:
                                colors.append(geom.colors[:, :3])
                elif isinstance(mesh, trimesh.points.PointCloud):
                    vertices.append(mesh.vertices)
                    if mesh.colors is not None:
                        colors.append(mesh.colors[:, :3])

                if not vertices:
                    return jsonify({"error": "No point data found in GLB"}), 400

                vertices = np.vstack(vertices)
                if colors:
                    colors = np.vstack(colors)
                else:
                    colors = np.ones_like(vertices) * 255

            # Normalize colors if needed
            if colors.max() <= 1.0:
                colors = (colors * 255.0).clip(0, 255)

            state.current_pointcloud = {
                "vertices": vertices.tolist(),
                "colors": colors.tolist(),
                "metadata": {
                    "num_points": len(vertices),
                    "filename": filename,
                    "source": "glb_upload"
                }
            }
            return jsonify({"status": "completed", "pointcloud": state.current_pointcloud})
        except Exception as e:
            print(f"❌ GLB/JSON ingest failed: {e}")
            return jsonify({"error": f"Failed to load point cloud: {e}"}), 500

    # Otherwise, require model readiness for inference
    if not state.model_ready:
        return jsonify({"error": "Model not loaded"}), 503

    # Get processing parameters
    resolution = int(request.form.get('resolution', 504))
    max_points = int(request.form.get('max_points', 1000000))

    # Process in background
    job_id = f"job_{int(time.time() * 1000)}"

    def process_async():
        try:
            # Check if file is video or image
            file_ext = filepath.suffix.lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

            if is_video:
                # Process video - extract frames
                from moviepy.editor import VideoFileClip

                clip = VideoFileClip(str(filepath))
                fps = clip.fps
                duration = clip.duration

                # Extract frames (sample every 0.5 seconds for performance)
                sample_rate = max(1, int(fps * 0.5))  # Sample every 0.5 seconds
                frames = []

                for i, frame in enumerate(clip.iter_frames()):
                    if i % sample_rate == 0:
                        frames.append(Image.fromarray(frame))
                    if len(frames) >= 20:  # Limit to 20 frames max
                        break

                clip.close()

                # Process frames with DA3
                prediction = state.model.inference(
                    image=frames,
                    process_res=resolution,
                    num_max_points=max_points
                )
            else:
                # Process single image
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
                    x3d = -(x_coords - cx) * depth / fx  # Negate X to match camera convention
                    y3d = -(y_coords - cy) * depth / fy  # Negate Y to convert from image coords (Y down) to 3D coords (Y up)
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
    try:
        vertices = np.array(state.current_pointcloud["vertices"], dtype=np.float32)
        colors = np.array(state.current_pointcloud["colors"], dtype=np.float32)

        # Normalize colors to 0-255 uint8 if needed
        if colors.max() <= 1.0:
            colors = (colors * 255.0).clip(0, 255)
        colors = colors.astype(np.uint8)

        # Build glb name from source filename
        src_name = state.current_pointcloud.get("metadata", {}).get("filename", "point_cloud")
        glb_name = f"{Path(src_name).stem}.glb"

        output_dir = Path(app.config['OUTPUT_FOLDER'])
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / glb_name

        # Create point cloud scene and export to GLB
        scene = trimesh.Scene()
        if len(vertices) > 0:
            scene.add_geometry(trimesh.points.PointCloud(vertices=vertices, colors=colors))
        scene.export(output_file, file_type='glb')

        return send_file(output_file, mimetype='model/gltf-binary', as_attachment=True, download_name=glb_name)
    except Exception as e:
        print(f"❌ GLB export failed: {e}")
        return jsonify({"error": f"Failed to export GLB: {e}"}), 500

# ============================================================================
# Ollama-style convenience endpoints (aliases for the above core routes)
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health():
    """Simple service heartbeat with model status."""
    return jsonify({
        "status": "ok",
        "model": {
            "id": state.current_model_id,
            "ready": state.model_ready,
            "loading": state.model_loading,
            "downloaded": state.current_model_id in state.downloaded_models
        },
        "active_jobs": len(state.processing_jobs)
    })

@app.route('/api/v1/models', methods=['GET'])
def v1_list_models():
    """Alias for list_models with simplified field names."""
    models = list_models().get_json()["models"]
    return jsonify({"data": models})

@app.route('/api/v1/models/download', methods=['POST'])
def v1_download_model():
    """
    Select and begin downloading/loading a model.
    Equivalent to calling /api/models/select then /api/load_model.
    """
    data = request.get_json()
    model_id = (data or {}).get("model_id")
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    ok, resp, code = _select_model_by_id(model_id)
    if not ok:
        return resp, code
    return load_model()

@app.route('/api/v1/models/load', methods=['POST'])
def v1_load_model():
    """Explicitly load a model by id."""
    data = request.get_json()
    model_id = (data or {}).get("model_id")
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    ok, resp, code = _select_model_by_id(model_id)
    if not ok:
        return resp, code
    return load_model()

@app.route('/api/v1/infer', methods=['POST'])
def v1_infer():
    """Alias for /api/process that accepts multipart form-data."""
    return process_file()

@app.route('/api/v1/jobs/<job_id>', methods=['GET'])
def v1_job_status(job_id):
    """Alias for job polling."""
    return get_job_status(job_id)

@app.route('/api/v1/export/glb', methods=['GET'])
def v1_export_glb():
    """Alias for GLB export."""
    return export_glb()

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
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/FlyControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/PointerLockControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
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
            <i class="fas fa-align-center"></i> Auto Floor
        </button>
        <button onclick="toggleManualFloorSelection()" id="manual-floor-btn" disabled>
            <i class="fas fa-mouse-pointer"></i> <span id="manual-floor-text">Select Floor</span>
        </button>
        <button onclick="toggleFPSMode()" id="fps-mode-btn" disabled>
            <i class="fas fa-running"></i> <span id="fps-mode-text">FPS Mode</span>
        </button>
        <button onclick="resetView()" id="reset-btn" disabled>
            <i class="fas fa-undo"></i> Reset View
        </button>
    </div>

    <!-- Pointer lock activation circle -->
    <div id="pointer-lock-indicator" style="
        position: fixed;
        pointer-events: none;
        display: none;
        z-index: 10000;
    ">
        <svg width="80" height="80" style="transform: translate(-40px, -40px);">
            <circle cx="40" cy="40" r="35" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="3"/>
            <circle id="progress-circle" cx="40" cy="40" r="35" fill="none" stroke="rgba(59, 130, 246, 0.8)"
                    stroke-width="3" stroke-dasharray="220" stroke-dashoffset="220"
                    style="transform: rotate(-90deg); transform-origin: 50% 50%; transition: stroke-dashoffset 0.1s linear;"/>
            <text x="40" y="45" text-anchor="middle" fill="white" font-size="12" font-family="Arial">Hold</text>
        </svg>
    </div>

    <!-- Pointer lock exit indicator -->
    <div id="pointer-lock-active" style="
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        display: none;
        z-index: 10000;
    ">
        <div style="
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.5);
            border-radius: 50%;
            background: rgba(255,255,255,0.1);
        "></div>
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
    <input type="file" id="file-input" accept="image/*,video/*,.glb,.gltf,.json" onchange="handleFileSelect(this.files[0])">

    <script>
        // Three.js setup
        let scene, camera, renderer, controls, pointerLockControls, pointCloud;
        let currentJobId = null;
        let currentModel = null;
        let manualFloorMode = false;
        let selectedPoints = [];
        let highlightedPoints = null;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();
        let originalPointCloudData = null;  // Store original point cloud data for color restoration

        // FPS mode with PointerLockControls
        let fpsModeActive = false;
        let pointerLocked = false;
        let holdStartTime = null;
        let holdTimeout = null;
        let holdAnimationFrame = null;
        let cameraYaw = 0;
        let cameraPitch = 0;
        const CAMERA_HEIGHT = 1.6;
        const WALK_SPEED = 3.5;
        const VERTICAL_SPEED = 2.0;
        const clock = new THREE.Clock();
        const tempEuler = new THREE.Euler(0, 0, 0, 'YXZ');  // used to zero roll in fly mode
        let moveForward = false;
        let moveBackward = false;
        let moveLeft = false;
        let moveRight = false;
        let moveUp = false;
        let moveDown = false;

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
            camera.position.set(0, 1.6, 5);  // Start at eye level (1.6m)

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // FlyControls (default free-fly camera)
            controls = new THREE.FlyControls(camera, renderer.domElement);
            controls.movementSpeed = 5;           // meters per second (scaled scene units)
            controls.rollSpeed = Math.PI / 8;     // softer roll feel
            controls.dragToLook = true;           // click-drag to pivot about camera position
            controls.autoForward = false;

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

            const delta = clock.getDelta();

            if (fpsModeActive && pointerLocked && pointerLockControls) {
                // FPS mode movement (WASD/Arrows with pointer lock)
                const moveStep = WALK_SPEED * delta;
                const verticalStep = VERTICAL_SPEED * delta;

                if (moveForward) pointerLockControls.moveForward(moveStep);
                if (moveBackward) pointerLockControls.moveForward(-moveStep);
                if (moveLeft) pointerLockControls.moveRight(-moveStep);
                if (moveRight) pointerLockControls.moveRight(moveStep);

                const rig = pointerLockControls.getObject();
                if (moveUp) rig.position.y += verticalStep;
                if (moveDown) rig.position.y -= verticalStep;
            } else {
                // Fly controls update
                if (controls && controls.enabled) {
                    controls.update(delta);
                    // Keep fly mode upright (remove roll)
                    tempEuler.setFromQuaternion(camera.quaternion, 'YXZ');
                    tempEuler.z = 0;
                    camera.quaternion.setFromEuler(tempEuler);
                    camera.up.set(0, 1, 0);
                }
            }

            renderer.render(scene, camera);
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function setupDragAndDrop() {
            // Drag and drop
            document.addEventListener('dragover', (e) => {
                e.preventDefault();
                const overlay = document.getElementById('drag-overlay');
                if (overlay) overlay.classList.add('active');
            });

            document.addEventListener('dragleave', (e) => {
                if (e.target === document.body || e.target === document.documentElement) {
                    const overlay = document.getElementById('drag-overlay');
                    if (overlay) overlay.classList.remove('active');
                }
            });

            document.addEventListener('drop', (e) => {
                e.preventDefault();
                const overlay = document.getElementById('drag-overlay');
                if (overlay) overlay.classList.remove('active');

                if (e.dataTransfer.files.length > 0) {
                    handleFileSelect(e.dataTransfer.files[0]);
                }
            });
        }

        function setupKeyboardControls() {
            // Keyboard movement state tracking
            document.addEventListener('keydown', (e) => {
                switch(e.key) {
                    case 'w':
                    case 'W':
                    case 'ArrowUp':
                        moveForward = true;
                        break;
                    case 's':
                    case 'S':
                    case 'ArrowDown':
                        moveBackward = true;
                        break;
                    case 'a':
                    case 'A':
                    case 'ArrowLeft':
                        moveLeft = true;
                        break;
                    case 'd':
                    case 'D':
                    case 'ArrowRight':
                        moveRight = true;
                        break;
                    case 'q':
                    case 'Q':
                        moveUp = true;
                        break;
                    case 'e':
                    case 'E':
                        moveDown = true;
                        break;
                    case ' ':
                        if (pointerLockControls && pointerLocked) {
                            pointerLockControls.getObject().position.y = CAMERA_HEIGHT;
                        } else {
                            camera.position.y = CAMERA_HEIGHT;
                        }
                        e.preventDefault();
                        break;
                }
            });

            document.addEventListener('keyup', (e) => {
                switch(e.key) {
                    case 'w':
                    case 'W':
                    case 'ArrowUp':
                        moveForward = false;
                        break;
                    case 's':
                    case 'S':
                    case 'ArrowDown':
                        moveBackward = false;
                        break;
                    case 'a':
                    case 'A':
                    case 'ArrowLeft':
                        moveLeft = false;
                        break;
                    case 'd':
                    case 'D':
                    case 'ArrowRight':
                        moveRight = false;
                        break;
                    case 'q':
                    case 'Q':
                        moveUp = false;
                        break;
                    case 'e':
                    case 'E':
                        moveDown = false;
                        break;
                }
            });

            console.log('Keyboard controls enabled: WASD/Arrows=Move, Q/E=Up/Down, Space=Reset height');
        }

        function ensurePointerLockControls() {
            if (pointerLockControls) return;

            // Capture current camera transform before reparenting into controls
            const worldPos = new THREE.Vector3();
            const worldDir = new THREE.Vector3();
            camera.getWorldPosition(worldPos);
            camera.getWorldDirection(worldDir);

            pointerLockControls = new THREE.PointerLockControls(camera, renderer.domElement);
            pointerLockControls.pointerSpeed = 0.35;

            const rig = pointerLockControls.getObject();
            scene.add(rig);

            pointerLockControls.addEventListener('lock', handlePointerLockEnter);
            pointerLockControls.addEventListener('unlock', handlePointerLockExit);

            // Initialize rig orientation to match current view
            camera.position.set(0, 0, 0);
            camera.rotation.set(0, 0, 0); // clear any roll from fly mode

            cameraYaw = Math.atan2(worldDir.x, worldDir.z);
            cameraPitch = Math.asin(-worldDir.y);

            rig.position.copy(worldPos);
            rig.position.y = CAMERA_HEIGHT; // force eye height for FPS entry
            rig.rotation.y = cameraYaw;

            const pitchObject = rig.children[0];
            if (pitchObject) {
                pitchObject.rotation.x = cameraPitch;
            }
        }

        function syncPointerLockRig() {
            if (!pointerLockControls) return;

            const rig = pointerLockControls.getObject();
            const pitchObject = rig.children[0];
            const worldPos = new THREE.Vector3();
            const worldDir = new THREE.Vector3();

            camera.getWorldPosition(worldPos);
            camera.getWorldDirection(worldDir);

            cameraYaw = Math.atan2(worldDir.x, worldDir.z);
            cameraPitch = Math.asin(-worldDir.y);

            rig.position.copy(worldPos);
            rig.position.y = CAMERA_HEIGHT; // reset to FPS eye height to avoid drift
            rig.rotation.y = cameraYaw;

            if (pitchObject) {
                pitchObject.rotation.x = cameraPitch;
                pitchObject.rotation.z = 0; // remove roll from fly controls
            }

            // Clear any residual roll on the camera itself
            camera.rotation.set(0, 0, 0);
        }

        function handlePointerLockEnter() {
            pointerLocked = true;
            document.getElementById('pointer-lock-active').style.display = 'block';

            if (controls) controls.enabled = false;
            document.getElementById('model-status-text').textContent = 'First-Person Mode (ESC to exit)';
            console.log('Pointer lock activated - WASD/Arrows to move, mouse to look, ESC to exit');
        }

        function restoreCameraAfterPointerLock() {
            if (!pointerLockControls) return;

            const rig = pointerLockControls.getObject();
            const pitchObject = rig.children[0];

            const worldPos = new THREE.Vector3();
            const worldQuat = new THREE.Quaternion();
            camera.getWorldPosition(worldPos);
            camera.getWorldQuaternion(worldQuat);

            if (pitchObject && pitchObject.children.includes(camera)) {
                pitchObject.remove(camera);
            }

            scene.add(camera);
            camera.position.copy(worldPos);
            camera.quaternion.copy(worldQuat);

            scene.remove(rig);
            pointerLockControls.removeEventListener('lock', handlePointerLockEnter);
            pointerLockControls.removeEventListener('unlock', handlePointerLockExit);
            pointerLockControls.dispose();
            pointerLockControls = null;

            if (controls) controls.enabled = true;
        }

        function handlePointerLockExit() {
            if (!pointerLocked && !pointerLockControls) return;

            pointerLocked = false;
            document.getElementById('pointer-lock-active').style.display = 'none';
            restoreCameraAfterPointerLock();

            if (fpsModeActive) {
                fpsModeActive = false;
                updateFPSButtonState();
            }

            document.getElementById('model-status-text').textContent = 'Model: Ready';
            console.log('Pointer lock deactivated');
        }

        function exitPointerLock() {
            if (document.pointerLockElement) {
                document.exitPointerLock();
            } else {
                handlePointerLockExit();
            }
        }

        function requestPointerLock() {
            ensurePointerLockControls();
            syncPointerLockRig();

            if (pointerLockControls) {
                // Reset local camera rotation to align with yaw/pitch-only FPS rig
                camera.rotation.set(0, 0, 0);
                pointerLockControls.lock();
            }
        }

        function setupPointerLockControls() {
            const indicator = document.getElementById('pointer-lock-indicator');
            const progressCircle = document.getElementById('progress-circle');
            const circumference = 220; // Match stroke-dasharray

            // Start hold on mousedown (not in floor selection mode)
            renderer.domElement.addEventListener('mousedown', (e) => {
                if (e.button === 0 && !manualFloorMode && !pointerLocked && fpsModeActive) {
                    holdStartTime = Date.now();

                    // Position indicator at cursor
                    indicator.style.left = e.clientX + 'px';
                    indicator.style.top = e.clientY + 'px';
                    indicator.style.display = 'block';

                    // Animate progress circle
                    const animateProgress = () => {
                        const elapsed = Date.now() - holdStartTime;
                        const progress = Math.min(elapsed / 3000, 1); // 3 seconds
                        const offset = circumference - (progress * circumference);
                        progressCircle.style.strokeDashoffset = offset;

                        if (progress < 1) {
                            holdAnimationFrame = requestAnimationFrame(animateProgress);
                        }
                    };
                    animateProgress();

                    // Request pointer lock after 3 seconds
                    holdTimeout = setTimeout(() => {
                        indicator.style.display = 'none';
                        requestPointerLock();
                    }, 3000);
                }
            });

            // Cancel hold on mouseup or mouseleave
            const cancelHold = () => {
                if (holdTimeout) {
                    clearTimeout(holdTimeout);
                    holdTimeout = null;
                }
                if (holdAnimationFrame) {
                    cancelAnimationFrame(holdAnimationFrame);
                    holdAnimationFrame = null;
                }
                indicator.style.display = 'none';
                progressCircle.style.strokeDashoffset = circumference;
            };

            renderer.domElement.addEventListener('mouseup', cancelHold);
            renderer.domElement.addEventListener('mouseleave', cancelHold);

            console.log('Pointer lock controls ready: Hold mouse button for 3 seconds to activate');
        }

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

        // File processing (images/videos go to backend; GLB/GLTF/JSON load locally)
        async function handleFileSelect(file) {
            if (!file) return;
            const lowerName = file.name.toLowerCase();
            const isLocalPointCloud = lowerName.endsWith('.glb') || lowerName.endsWith('.gltf') || lowerName.endsWith('.json');

            if (isLocalPointCloud) {
                loadLocalGLB(file);
                return;
            }

            // Otherwise: send to backend for inference
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

                if (response.ok && data.pointcloud) {
                    // Direct load (e.g., server-side GLB/JSON upload bypassing inference)
                    loadPointCloud(data.pointcloud);
                    document.getElementById('model-status-text').textContent = 'Model: Ready';
                    document.getElementById('export-btn').disabled = false;
                    document.getElementById('floor-btn').disabled = false;
                    document.getElementById('manual-floor-btn').disabled = false;
                    document.getElementById('fps-mode-btn').disabled = false;
                    document.getElementById('reset-btn').disabled = false;
                    return;
                } else if (response.ok && data.job_id) {
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

        function loadLocalGLB(file) {
            document.getElementById('model-status-text').textContent = 'Loading local point cloud...';
            const loader = new THREE.GLTFLoader();
            const url = URL.createObjectURL(file);

            loader.load(url, (gltf) => {
                // Remove existing point cloud
                if (pointCloud) {
                    scene.remove(pointCloud);
                    pointCloud = null;
                }

                let loadedPoints = null;
                gltf.scene.traverse((child) => {
                    if (child.isPoints) {
                        const clone = child.clone();
                        if (clone.material && clone.material.size === undefined) {
                            clone.material.size = 0.01;
                        }
                        loadedPoints = loadedPoints || clone;
                    } else if (child.isMesh) {
                        const geom = child.geometry;
                        if (!geom) return;
                        const hasColor = !!geom.attributes.color;
                        const mat = new THREE.PointsMaterial({ size: 0.01, vertexColors: hasColor, color: hasColor ? undefined : 0xffffff });
                        const pts = new THREE.Points(geom, mat);
                        loadedPoints = loadedPoints || pts;
                    }
                });

                if (!loadedPoints) {
                    // Fallback: use the full scene
                    loadedPoints = gltf.scene;
                }

                pointCloud = loadedPoints;
                scene.add(pointCloud);

                // Update UI/state
                // Try to count points if geometry exists
                let numPoints = 0;
                if (pointCloud.isPoints && pointCloud.geometry && pointCloud.geometry.attributes.position) {
                    numPoints = pointCloud.geometry.attributes.position.count;
                }
                document.getElementById('points-count').textContent = `Points: ${numPoints.toLocaleString()}`;
                document.getElementById('model-status-text').textContent = 'Model: Ready';
                document.getElementById('export-btn').disabled = false;
                document.getElementById('floor-btn').disabled = false;
                document.getElementById('manual-floor-btn').disabled = false;
                document.getElementById('fps-mode-btn').disabled = false;
                document.getElementById('reset-btn').disabled = false;

                URL.revokeObjectURL(url);
            }, undefined, (err) => {
                console.error('GLB load error:', err);
                alert('Failed to load GLB. See console for details.');
                document.getElementById('model-status-text').textContent = 'Model: Ready';
                URL.revokeObjectURL(url);
            });
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
                        document.getElementById('manual-floor-btn').disabled = false;
                        document.getElementById('fps-mode-btn').disabled = false;
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

                // Store original data for color restoration
                originalPointCloudData = {
                    vertices: data.vertices,
                    colors: data.colors,
                    metadata: data.metadata
                };

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
                camera.lookAt(center);
                if (controls) controls.update(0);

                console.log('Point cloud loaded successfully');
            } catch (error) {
                console.error('Error loading point cloud:', error);
                alert('Failed to load point cloud. Check console for details.');
            }
        }

        // Floor alignment (automatic)
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

        // Manual floor selection
        function toggleManualFloorSelection() {
            manualFloorMode = !manualFloorMode;
            const btn = document.getElementById('manual-floor-btn');
            const text = document.getElementById('manual-floor-text');

            if (manualFloorMode) {
                btn.style.background = 'rgba(59, 130, 246, 0.5)';
                btn.style.borderColor = 'rgba(59, 130, 246, 0.8)';
                text.textContent = 'Click Floor';
                document.getElementById('model-status-text').textContent = 'Click on floor surface...';
                console.log('Manual floor selection mode: ON');
            } else {
                btn.style.background = 'rgba(16, 185, 129, 0.2)';
                btn.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                text.textContent = 'Select Floor';
                document.getElementById('model-status-text').textContent = 'Model: Ready';
                clearHighlightedPoints();
                selectedPoints = [];
                console.log('Manual floor selection mode: OFF');
            }
        }

        function onMouseClick(event) {
            if (!manualFloorMode || !pointCloud) return;

            // Calculate mouse position in normalized device coordinates
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            // Update raycaster
            raycaster.setFromCamera(mouse, camera);

            // Check for intersections
            const intersects = raycaster.intersectObject(pointCloud, true);

            if (intersects.length > 0) {
                const intersect = intersects[0];
                const point = intersect.point;

                console.log('Clicked point:', point);

                // Find nearby points
                findNearbyPoints(point);

                // Fit plane and align
                if (selectedPoints.length > 100) {
                    applyManualFloorAlignment();
                }
            }
        }

        function findNearbyPoints(clickedPoint) {
            if (!pointCloud) return;

            const positions = pointCloud.geometry.attributes.position.array;
            const colors = pointCloud.geometry.attributes.color.array;

            const searchRadius = 0.5;  // 0.5 meter radius
            const nearby = [];
            const nearbyIndices = [];

            // Find all points within radius
            for (let i = 0; i < positions.length; i += 3) {
                const x = positions[i];
                const y = positions[i + 1];
                const z = positions[i + 2];

                const dx = x - clickedPoint.x;
                const dy = y - clickedPoint.y;
                const dz = z - clickedPoint.z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (distance < searchRadius) {
                    nearby.push(new THREE.Vector3(x, y, z));
                    nearbyIndices.push(i / 3);
                }
            }

            selectedPoints = nearby;

            console.log(`Found ${nearby.length} nearby points within ${searchRadius}m`);

            // Highlight selected points
            highlightPoints(nearbyIndices, colors);
        }

        function highlightPoints(indices, originalColors) {
            if (!pointCloud) return;

            const colors = pointCloud.geometry.attributes.color.array;

            // Reset all colors first
            for (let i = 0; i < colors.length; i++) {
                colors[i] = originalColors[i];
            }

            // Highlight selected points (white)
            indices.forEach(idx => {
                colors[idx * 3] = 1.0;      // R
                colors[idx * 3 + 1] = 1.0;  // G
                colors[idx * 3 + 2] = 1.0;  // B
            });

            pointCloud.geometry.attributes.color.needsUpdate = true;

            document.getElementById('model-status-text').textContent =
                `Selected ${indices.length} points. Processing...`;
        }

        function clearHighlightedPoints() {
            if (!pointCloud || !originalPointCloudData) return;

            const colors = pointCloud.geometry.attributes.color.array;
            const originalColors = originalPointCloudData.colors;

            // Restore original colors
            for (let i = 0; i < originalColors.length; i++) {
                for (let j = 0; j < 3; j++) {
                    colors[i * 3 + j] = originalColors[i][j] / 255;
                }
            }

            pointCloud.geometry.attributes.color.needsUpdate = true;
            console.log('Original colors restored');
        }

        function fitPlaneToPoints(points) {
            // Fit plane using least squares
            // Plane equation: ax + by + cz + d = 0

            const n = points.length;
            if (n < 3) return null;

            // Calculate centroid
            const centroid = new THREE.Vector3();
            points.forEach(p => centroid.add(p));
            centroid.divideScalar(n);

            // Build covariance matrix
            let xx = 0, xy = 0, xz = 0;
            let yy = 0, yz = 0, zz = 0;

            points.forEach(p => {
                const dx = p.x - centroid.x;
                const dy = p.y - centroid.y;
                const dz = p.z - centroid.z;

                xx += dx * dx;
                xy += dx * dy;
                xz += dx * dz;
                yy += dy * dy;
                yz += dy * dz;
                zz += dz * dz;
            });

            // Find eigenvector corresponding to smallest eigenvalue
            // Simplified: assume normal is mostly vertical
            // Use cross product of two vectors in the plane

            // Get two vectors from centroid to sample points
            const v1 = new THREE.Vector3().subVectors(points[0], centroid);
            const v2 = new THREE.Vector3().subVectors(points[Math.floor(n/2)], centroid);

            // Normal is cross product
            const normal = new THREE.Vector3().crossVectors(v1, v2);
            normal.normalize();

            // Ensure normal points upward
            if (normal.y < 0) {
                normal.multiplyScalar(-1);
            }

            // Calculate d coefficient
            const d = -normal.dot(centroid);

            return { normal, d, centroid };
        }

        function applyManualFloorAlignment() {
            if (selectedPoints.length < 100) {
                alert('Not enough points selected. Click on a larger floor area.');
                return;
            }

            document.getElementById('model-status-text').textContent = 'Fitting plane...';

            // Fit plane to selected points
            const plane = fitPlaneToPoints(selectedPoints);

            if (!plane) {
                alert('Failed to fit plane to selected points.');
                document.getElementById('model-status-text').textContent = 'Model: Ready';
                return;
            }

            console.log('Fitted plane normal:', plane.normal);
            console.log('Plane centroid:', plane.centroid);

            // Create rotation to align plane with XZ (y=0)
            const targetNormal = new THREE.Vector3(0, 1, 0);
            const quaternion = new THREE.Quaternion();
            quaternion.setFromUnitVectors(plane.normal, targetNormal);

            // Apply rotation to all points
            const positions = pointCloud.geometry.attributes.position.array;
            const rotationMatrix = new THREE.Matrix4().makeRotationFromQuaternion(quaternion);

            for (let i = 0; i < positions.length; i += 3) {
                const point = new THREE.Vector3(
                    positions[i],
                    positions[i + 1],
                    positions[i + 2]
                );

                point.applyMatrix4(rotationMatrix);

                positions[i] = point.x;
                positions[i + 1] = point.y;
                positions[i + 2] = point.z;
            }

            // Find lowest point and translate to y=0
            let minY = Infinity;
            for (let i = 1; i < positions.length; i += 3) {
                if (positions[i] < minY) minY = positions[i];
            }

            for (let i = 1; i < positions.length; i += 3) {
                positions[i] -= minY;
            }

            pointCloud.geometry.attributes.position.needsUpdate = true;
            pointCloud.geometry.computeBoundingSphere();

            // Update stored point cloud data with new aligned vertices
            const newVertices = [];
            for (let i = 0; i < positions.length; i += 3) {
                newVertices.push([positions[i], positions[i + 1], positions[i + 2]]);
            }

            if (originalPointCloudData) {
                originalPointCloudData.vertices = newVertices;
            }

            // Exit selection mode (this will restore original colors)
            toggleManualFloorSelection();

            alert(`Floor aligned! Used ${selectedPoints.length} points for plane fitting.`);
            document.getElementById('model-status-text').textContent = 'Model: Ready';

            console.log('Manual floor alignment complete');
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
            camera.position.set(0, CAMERA_HEIGHT, 5);  // Reset to eye level
            camera.lookAt(new THREE.Vector3(0, CAMERA_HEIGHT, 0));
            if (controls) controls.update(0);
        }

        // Toggle FPS mode with dolly camera
        function toggleFPSMode() {
            fpsModeActive = !fpsModeActive;
            updateFPSButtonState();

            if (fpsModeActive) {
                requestPointerLock();
                console.log('FPS Mode activated');
            } else {
                exitPointerLock();
                console.log('FPS Mode deactivated');
            }
        }

        function updateFPSButtonState() {
            const btn = document.getElementById('fps-mode-btn');
            const text = document.getElementById('fps-mode-text');

            if (fpsModeActive) {
                btn.style.background = 'rgba(59, 130, 246, 0.5)';
                btn.style.borderColor = 'rgba(59, 130, 246, 0.8)';
                text.textContent = 'Exit FPS';
            } else {
                btn.style.background = 'rgba(16, 185, 129, 0.2)';
                btn.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                text.textContent = 'FPS Mode';
            }
        }

        function setupModalCloseHandler() {
            // Close modal when clicking outside
            window.addEventListener('click', (e) => {
                const modal = document.getElementById('model-modal');
                if (modal && e.target === modal) {
                    closeModelSelector();
                }
            });
        }

        function setupManualFloorSelection() {
            // Add click listener for manual floor selection
            if (renderer && renderer.domElement) {
                renderer.domElement.addEventListener('click', onMouseClick, false);
                console.log('Manual floor selection click handler registered');
            }
        }

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {
            console.log('Initializing Depth Anything 3 UI...');

            // Initialize Three.js scene
            initThreeJS();
            console.log('Three.js scene initialized');

            // Setup event handlers
            setupDragAndDrop();
            setupModalCloseHandler();
            setupKeyboardControls();
            setupPointerLockControls();
            setupManualFloorSelection();
            console.log('Event handlers setup complete');

            // Update model status
            updateModelStatus();
            console.log('UI initialization complete');
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
