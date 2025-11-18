#!/usr/bin/env python3
"""
Depth Anything 3 - Standalone Flask Application
Self-bootstrapping application with venv setup, dependency installation,
and complete web interface for video processing and point cloud visualization.
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
            # Install it BEFORE the package to avoid dependency conflicts
            print("\n📦 Installing xformers (optional)...")
            print("   ℹ️  If this fails, the app will still work (see README FAQ)")
            try:
                # Install xformers with pip (now torch is already installed)
                subprocess.run([str(venv_pip), "install", "xformers"],
                             check=True, timeout=300)
                print("✓ xformers installed successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print("⚠️  xformers installation failed or timed out")
                print("   The application will work without it (with slightly reduced performance)")
                print("   For older GPU support, see: https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11")

            # Install the package itself WITHOUT dependencies (we already installed them)
            print("\n📦 Installing depth-anything-3 package (without dependencies)...")
            subprocess.run([str(venv_pip), "install", "-e", ".", "--no-deps"], check=True)

            # Ensure moviepy is installed (needed for video processing)
            print("\n📦 Ensuring moviepy is installed...")
            try:
                subprocess.run([str(venv_pip), "install", "moviepy==1.0.3"], check=True)
                print("✓ moviepy installed")
            except subprocess.CalledProcessError:
                print("⚠️  Warning: moviepy installation had issues, trying without version pin...")
                subprocess.run([str(venv_pip), "install", "moviepy"], check=True)

            # Install Flask (not in requirements.txt)
            print("\n📦 Installing Flask...")
            subprocess.run([str(venv_pip), "install", "flask", "flask-cors"], check=True)

            # Create marker file
            marker_file.touch()
            print("\n✓ Installation complete!")
            print("   Note: Some optional packages may have been skipped")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install dependencies: {e}")
            print("   Try manually installing with: pip install -r requirements.txt")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during installation: {e}")
            return False
    else:
        print("✓ Dependencies already installed")

    # Restart in venv if not already there
    print("\n🔄 Restarting in virtual environment...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Run bootstrap if needed
if __name__ == "__main__" and not (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    bootstrap_environment()
    sys.exit(0)

# ============================================================================
# MAIN APPLICATION - Only runs after bootstrap
# ============================================================================

from flask import Flask, render_template_string, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import torch
from datetime import datetime
import glob
import shutil
import base64
import io
from PIL import Image
import socket

# Import DA3 modules (after bootstrap)
try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.app.css_and_html import GRADIO_CSS, get_header_html, get_description_html
except ImportError as e:
    print(f"❌ Error importing DA3 modules: {e}")
    print("Make sure the virtual environment is activated and dependencies are installed.")
    sys.exit(1)

# ============================================================================
# FLASK APPLICATION SETUP
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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = 'depth-anything-3-secret-key'

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Global state for model and processing
class AppState:
    def __init__(self):
        self.model = None
        self.model_loading = False
        self.model_ready = False
        self.model_name = "depth-anything/DA3NESTED-GIANT-LARGE"
        self.processing_jobs = {}
        self.lock = threading.Lock()

state = AppState()

# ============================================================================
# REGISTER API ENDPOINTS
# ============================================================================

# Import and register RESTful API endpoints
try:
    from api_endpoints import create_api_routes
    create_api_routes(app, state)
except ImportError as e:
    print(f"⚠️  Warning: Could not load API endpoints: {e}")


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 - Video to Point Cloud</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: #ffffff;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 40px 20px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
            border-radius: 15px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .header h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 10px;
        }

        .status-panel {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .status-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }

        .status-icon {
            font-size: 24px;
            margin-right: 15px;
            min-width: 30px;
        }

        .status-ready { color: #10b981; }
        .status-loading { color: #f59e0b; }
        .status-error { color: #ef4444; }

        .upload-section {
            background: rgba(0, 0, 0, 0.3);
            border: 2px dashed rgba(59, 130, 246, 0.5);
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-section:hover {
            border-color: rgba(59, 130, 246, 0.8);
            background: rgba(59, 130, 246, 0.05);
        }

        .upload-section.dragover {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }

        .btn {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 5px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        .viewer-container {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }

        #viewer {
            width: 100%;
            height: 600px;
            background: #000;
            border-radius: 10px;
        }

        .log-container {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }

        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #3b82f6;
            padding-left: 10px;
        }

        .settings-panel {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .setting-item {
            margin: 15px 0;
        }

        .setting-item label {
            display: block;
            margin-bottom: 5px;
            color: #cbd5e1;
        }

        .setting-item input,
        .setting-item select {
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 5px;
            color: white;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 3px solid #3b82f6;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-cube"></i> Depth Anything 3</h1>
            <p style="font-size: 1.2em; margin-top: 10px;">Video to 3D Point Cloud Generator</p>
        </div>

        <!-- Model Status Panel -->
        <div class="status-panel">
            <h2 style="margin-bottom: 15px;"><i class="fas fa-server"></i> System Status</h2>
            <div class="status-item">
                <i class="fas fa-brain status-icon" id="model-icon"></i>
                <div style="flex: 1;">
                    <strong>Model Status:</strong>
                    <span id="model-status">Initializing...</span>
                    <div id="model-progress" style="display: none;">
                        <div class="progress-bar" style="margin-top: 10px;">
                            <div class="progress-fill" id="model-progress-fill" style="width: 0%">0%</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="status-item">
                <i class="fas fa-microchip status-icon" id="device-icon"></i>
                <div>
                    <strong>Device:</strong>
                    <span id="device-status">Detecting...</span>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <!-- Upload and Settings -->
            <div>
                <!-- Upload Section -->
                <div class="upload-section" id="upload-area">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 48px; color: #3b82f6; margin-bottom: 20px;"></i>
                    <h3>Upload Video or Images</h3>
                    <p style="margin: 10px 0; color: #94a3b8;">Drag & drop or click to select</p>
                    <p style="font-size: 14px; color: #64748b;">Supported: MP4, AVI, MOV, PNG, JPG</p>
                    <input type="file" id="file-input" style="display: none;" accept="video/*,image/*" multiple>
                    <div id="file-info" style="margin-top: 15px; color: #3b82f6;"></div>
                </div>

                <!-- Settings Panel -->
                <div class="settings-panel">
                    <h3 style="margin-bottom: 15px;"><i class="fas fa-sliders-h"></i> Processing Settings</h3>

                    <div class="setting-item">
                        <label><i class="fas fa-film"></i> Video FPS (frames/second)</label>
                        <input type="number" id="fps" value="2" min="0.5" max="30" step="0.5">
                    </div>

                    <div class="setting-item">
                        <label><i class="fas fa-expand-arrows-alt"></i> Processing Resolution</label>
                        <select id="resolution">
                            <option value="504">504px (Fast)</option>
                            <option value="756">756px (Balanced)</option>
                            <option value="1024">1024px (Quality)</option>
                        </select>
                    </div>

                    <div class="setting-item">
                        <label><i class="fas fa-layer-group"></i> Export Format</label>
                        <select id="export-format">
                            <option value="glb">GLB (3D Point Cloud)</option>
                            <option value="glb-depth_vis">GLB + Depth Visualization</option>
                            <option value="mini_npz-glb">NPZ + GLB</option>
                            <option value="glb-feat_vis">GLB + Feature Visualization</option>
                        </select>
                    </div>

                    <div class="setting-item">
                        <label><i class="fas fa-braille"></i> Max Points in Cloud</label>
                        <input type="number" id="max-points" value="1000000" min="100000" max="5000000" step="100000">
                    </div>

                    <button class="btn" id="process-btn" onclick="processVideo()" disabled>
                        <i class="fas fa-play"></i> Start Processing
                    </button>
                    <button class="btn" id="load-model-btn" onclick="loadModel()">
                        <i class="fas fa-download"></i> Load Model
                    </button>
                </div>

                <!-- Processing Status -->
                <div class="status-panel" id="processing-panel" style="display: none;">
                    <h3><i class="fas fa-cog fa-spin"></i> Processing</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" id="processing-progress-fill" style="width: 0%">0%</div>
                    </div>
                    <p id="processing-status" style="margin-top: 10px; text-align: center;">Initializing...</p>
                </div>
            </div>

            <!-- 3D Viewer -->
            <div>
                <div class="viewer-container">
                    <h3 style="margin-bottom: 15px;">
                        <i class="fas fa-cube"></i> 3D Point Cloud Viewer
                        <button class="btn" onclick="resetCamera()" style="float: right; padding: 8px 15px;">
                            <i class="fas fa-sync-alt"></i> Reset View
                        </button>
                    </h3>
                    <div id="viewer"></div>
                    <p style="margin-top: 10px; text-align: center; color: #94a3b8; font-size: 14px;">
                        <i class="fas fa-mouse"></i> Left-click: Rotate | Right-click: Pan | Scroll: Zoom
                    </p>
                </div>

                <!-- Log Container -->
                <div class="log-container" id="log-container">
                    <div class="log-entry">System initialized. Ready to process videos.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let currentModel = null;
        let statusCheckInterval = null;

        // Initialize Three.js viewer
        function initViewer() {
            const container = document.getElementById('viewer');

            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);

            // Camera
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 2, 5);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
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
            const gridHelper = new THREE.GridHelper(10, 10);
            scene.add(gridHelper);

            // Axes
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);

            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();

            // Handle window resize
            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            addLog('3D Viewer initialized');
        }

        // Load GLB model
        function loadGLB(url) {
            const loader = new THREE.GLTFLoader();

            // Remove previous model
            if (currentModel) {
                scene.remove(currentModel);
            }

            loader.load(url, (gltf) => {
                currentModel = gltf.scene;
                scene.add(currentModel);

                // Center and scale the model
                const box = new THREE.Box3().setFromObject(currentModel);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                currentModel.position.x = -center.x;
                currentModel.position.y = -center.y;
                currentModel.position.z = -center.z;

                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 5 / maxDim;
                currentModel.scale.set(scale, scale, scale);

                camera.position.set(0, 2, 8);
                controls.target.set(0, 0, 0);
                controls.update();

                addLog('Point cloud loaded successfully');
            }, undefined, (error) => {
                addLog('Error loading point cloud: ' + error, 'error');
            });
        }

        function resetCamera() {
            camera.position.set(0, 2, 5);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        // File upload handling
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            const files = fileInput.files;
            if (files.length > 0) {
                const file = files[0];
                fileInfo.innerHTML = `<i class="fas fa-check-circle"></i> Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                document.getElementById('process-btn').disabled = !state.modelReady;
                addLog(`File selected: ${file.name}`);
            }
        }

        // Load model
        async function loadModel() {
            const btn = document.getElementById('load-model-btn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Model...';

            document.getElementById('model-progress').style.display = 'block';
            addLog('Requesting model load...');

            try {
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const data = await response.json();

                if (data.status === 'loading') {
                    addLog('Model loading started...');
                    startStatusPolling();
                } else if (data.status === 'ready') {
                    updateModelStatus('ready');
                    addLog('Model is already loaded');
                }
            } catch (error) {
                addLog('Error loading model: ' + error, 'error');
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-download"></i> Load Model';
            }
        }

        // Process video
        async function processVideo() {
            const files = fileInput.files;
            if (files.length === 0) {
                alert('Please select a file first');
                return;
            }

            const formData = new FormData();
            formData.append('file', files[0]);
            formData.append('fps', document.getElementById('fps').value);
            formData.append('resolution', document.getElementById('resolution').value);
            formData.append('export_format', document.getElementById('export-format').value);
            formData.append('max_points', document.getElementById('max-points').value);

            document.getElementById('processing-panel').style.display = 'block';
            document.getElementById('process-btn').disabled = true;

            addLog('Starting video processing...');

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.job_id) {
                    addLog(`Job created: ${data.job_id}`);
                    pollJobStatus(data.job_id);
                } else {
                    addLog('Error: No job ID received', 'error');
                    document.getElementById('process-btn').disabled = false;
                }
            } catch (error) {
                addLog('Error starting process: ' + error, 'error');
                document.getElementById('process-btn').disabled = false;
            }
        }

        // Poll job status
        function pollJobStatus(jobId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/status/${jobId}`);
                    const data = await response.json();

                    updateProcessingProgress(data);

                    if (data.status === 'completed') {
                        clearInterval(interval);
                        handleJobComplete(data);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        handleJobError(data);
                    }
                } catch (error) {
                    addLog('Error polling status: ' + error, 'error');
                }
            }, 1000);
        }

        function updateProcessingProgress(data) {
            const progressBar = document.getElementById('processing-progress-fill');
            const statusText = document.getElementById('processing-status');

            progressBar.style.width = data.progress + '%';
            progressBar.textContent = data.progress + '%';
            statusText.textContent = data.message;

            if (data.log) {
                addLog(data.log);
            }
        }

        function handleJobComplete(data) {
            addLog('Processing completed!', 'success');
            document.getElementById('processing-panel').style.display = 'none';
            document.getElementById('process-btn').disabled = false;

            if (data.glb_url) {
                loadGLB(data.glb_url);
                addLog('Loading point cloud into viewer...');
            }

            if (data.outputs) {
                addLog(`Output files: ${data.outputs.join(', ')}`);
            }
        }

        function handleJobError(data) {
            addLog('Processing failed: ' + data.message, 'error');
            document.getElementById('processing-panel').style.display = 'none';
            document.getElementById('process-btn').disabled = false;
        }

        // Status polling for model loading
        function startStatusPolling() {
            if (statusCheckInterval) return;

            statusCheckInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/model_status');
                    const data = await response.json();

                    updateModelStatus(data.status, data.progress);

                    if (data.status === 'ready') {
                        clearInterval(statusCheckInterval);
                        statusCheckInterval = null;
                    }
                } catch (error) {
                    console.error('Error checking status:', error);
                }
            }, 500);
        }

        function updateModelStatus(status, progress = 0) {
            const icon = document.getElementById('model-icon');
            const statusText = document.getElementById('model-status');
            const progressDiv = document.getElementById('model-progress');
            const progressBar = document.getElementById('model-progress-fill');
            const loadBtn = document.getElementById('load-model-btn');
            const processBtn = document.getElementById('process-btn');

            if (status === 'ready') {
                icon.className = 'fas fa-brain status-icon status-ready';
                statusText.innerHTML = '<span class="status-ready">Ready</span>';
                progressDiv.style.display = 'none';
                loadBtn.style.display = 'none';
                processBtn.disabled = fileInput.files.length === 0;
                state.modelReady = true;
            } else if (status === 'loading') {
                icon.className = 'fas fa-brain status-icon status-loading';
                statusText.innerHTML = '<span class="status-loading">Loading...</span>';
                progressDiv.style.display = 'block';
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
                state.modelReady = false;
            } else {
                icon.className = 'fas fa-brain status-icon';
                statusText.textContent = 'Not loaded';
                state.modelReady = false;
            }
        }

        function addLog(message, type = 'info') {
            const logContainer = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';

            const timestamp = new Date().toLocaleTimeString();
            let icon = 'fa-info-circle';
            let color = '#3b82f6';

            if (type === 'success') {
                icon = 'fa-check-circle';
                color = '#10b981';
            } else if (type === 'error') {
                icon = 'fa-exclamation-circle';
                color = '#ef4444';
            }

            entry.innerHTML = `<i class="fas ${icon}" style="color: ${color}; margin-right: 8px;"></i>[${timestamp}] ${message}`;
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }

        // State object
        const state = {
            modelReady: false
        };

        // Initialize on load
        window.addEventListener('load', () => {
            initViewer();
            checkInitialStatus();
        });

        async function checkInitialStatus() {
            try {
                const response = await fetch('/api/model_status');
                const data = await response.json();
                updateModelStatus(data.status);

                // Check device
                const deviceResponse = await fetch('/api/device_info');
                const deviceData = await deviceResponse.json();
                document.getElementById('device-status').innerHTML =
                    `<span style="color: ${deviceData.cuda ? '#10b981' : '#f59e0b'}">${deviceData.device}</span>`;
                document.getElementById('device-icon').className =
                    `fas fa-microchip status-icon ${deviceData.cuda ? 'status-ready' : 'status-loading'}`;
            } catch (error) {
                addLog('Error checking initial status: ' + error, 'error');
            }
        }
    </script>
</body>
</html>
"""

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/model_status')
def model_status():
    """Get model loading status."""
    if state.model_ready:
        return jsonify({'status': 'ready', 'progress': 100})
    elif state.model_loading:
        return jsonify({'status': 'loading', 'progress': 50})
    else:
        return jsonify({'status': 'not_loaded', 'progress': 0})

@app.route('/api/device_info')
def device_info():
    """Get device information."""
    cuda_available = torch.cuda.is_available()
    device = "CUDA (GPU)" if cuda_available else "CPU"
    return jsonify({
        'device': device,
        'cuda': cuda_available
    })

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load the DA3 model."""
    if state.model_ready:
        return jsonify({'status': 'ready', 'message': 'Model already loaded'})

    if state.model_loading:
        return jsonify({'status': 'loading', 'message': 'Model is currently loading'})

    def load_model_thread():
        with state.lock:
            state.model_loading = True

        try:
            print("Loading Depth Anything 3 model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DepthAnything3.from_pretrained(state.model_name)
            model = model.to(device=device)
            model.eval()

            with state.lock:
                state.model = model
                state.model_ready = True
                state.model_loading = False

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            with state.lock:
                state.model_loading = False

    thread = threading.Thread(target=load_model_thread)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'loading', 'message': 'Model loading started'})

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process uploaded video/images."""
    if not state.model_ready:
        return jsonify({'error': 'Model not loaded'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = f"{timestamp}_{filename}"

    upload_path = Path(app.config['UPLOAD_FOLDER']) / job_id
    upload_path.mkdir(exist_ok=True)

    file_path = upload_path / filename
    file.save(str(file_path))

    # Get settings
    fps = float(request.form.get('fps', 2.0))
    resolution = int(request.form.get('resolution', 504))
    export_format = request.form.get('export_format', 'glb')
    max_points = int(request.form.get('max_points', 1000000))

    # Create job
    output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
    output_dir.mkdir(exist_ok=True)

    job_info = {
        'job_id': job_id,
        'status': 'processing',
        'progress': 0,
        'message': 'Starting processing...',
        'file_path': str(file_path),
        'output_dir': str(output_dir),
        'settings': {
            'fps': fps,
            'resolution': resolution,
            'export_format': export_format,
            'max_points': max_points
        }
    }

    state.processing_jobs[job_id] = job_info

    # Start processing in background thread
    def process_thread():
        try:
            job_info['message'] = 'Extracting frames...'
            job_info['progress'] = 10

            # Extract frames from video
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                frames_dir = output_dir / 'frames'
                frames_dir.mkdir(exist_ok=True)

                # Use moviepy to extract frames
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(str(file_path))
                duration = clip.duration
                frame_times = np.arange(0, duration, 1.0 / fps)

                for i, t in enumerate(frame_times):
                    frame = clip.get_frame(t)
                    img = Image.fromarray(frame)
                    img.save(frames_dir / f"frame_{i:04d}.png")

                clip.close()

                image_files = sorted(glob.glob(str(frames_dir / "*.png")))
            else:
                # Single image
                image_files = [str(file_path)]

            job_info['message'] = f'Processing {len(image_files)} frames...'
            job_info['progress'] = 30

            # Run inference
            prediction = state.model.inference(
                image=image_files,
                export_dir=str(output_dir),
                export_format=export_format,
                process_res=resolution,
                num_max_points=max_points,
                show_cameras=True
            )

            job_info['message'] = 'Generating outputs...'
            job_info['progress'] = 90

            # Find generated files
            glb_file = output_dir / 'scene.glb'
            outputs = []

            if glb_file.exists():
                outputs.append('scene.glb')
                job_info['glb_url'] = f'/api/output/{job_id}/scene.glb'

            for f in output_dir.glob('*'):
                if f.is_file() and f.name != filename:
                    outputs.append(f.name)

            job_info['status'] = 'completed'
            job_info['progress'] = 100
            job_info['message'] = 'Processing completed!'
            job_info['outputs'] = outputs

        except Exception as e:
            job_info['status'] = 'error'
            job_info['message'] = f'Error: {str(e)}'
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'processing'})

@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Get processing job status."""
    if job_id not in state.processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = state.processing_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'glb_url': job.get('glb_url'),
        'outputs': job.get('outputs', [])
    })

@app.route('/api/output/<job_id>/<filename>')
def serve_output(job_id, filename):
    """Serve output files."""
    output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
    return send_from_directory(output_dir, filename)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Find available port
    try:
        port = find_available_port(start_port=5000, max_attempts=10)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 Depth Anything 3 - Flask Application")
    print("=" * 70)
    print(f"✓ Virtual environment: Active")
    print(f"✓ Dependencies: Installed")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print("=" * 70)
    print("\n📡 Starting Flask server...")
    print(f"🌐 Open your browser and navigate to: http://localhost:{port}")
    if port != 5000:
        print(f"   ℹ️  Note: Using port {port} (port 5000 was in use)")
    print("\n💡 Tips:")
    print("   - Click 'Load Model' to download and load the DA3 model (first time only)")
    print("   - Upload a video or images using the upload area")
    print("   - Adjust settings and click 'Start Processing'")
    print("   - View the 3D point cloud in the interactive viewer")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
