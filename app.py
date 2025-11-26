#!/usr/bin/env python3
"""
Depth Anything 3 - API-only Flask Application
This server exposes the REST endpoints without the bundled Three.js frontend.
"""

import os
import sys
import subprocess
import json
import time
import threading
import signal
import platform
import re
import urllib.request
from pathlib import Path

# Try to import packaging, install if missing (needed for bootstrap)
try:
    from packaging.version import Version
except ImportError:
    print("📦 Installing packaging module (needed for bootstrap)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "packaging"], check=False, capture_output=True)
    try:
        from packaging.version import Version
    except ImportError:
        # Fallback: use a simple version comparison
        class Version:
            def __init__(self, v):
                self.v = tuple(int(x) if x.isdigit() else x for x in v.replace('a', '.').replace('b', '.').replace('rc', '.').split('.'))
            def __gt__(self, other):
                return self.v > other.v
            def __lt__(self, other):
                return self.v < other.v
            def __eq__(self, other):
                return self.v == other.v

# ============================================================================
# BOOTSTRAP SECTION - Virtual Environment Setup
# ============================================================================

def _is_jetson():
    """Return True if running on a Jetson (aarch64 + nv_tegra_release present)."""
    try:
        return platform.machine().lower() == "aarch64" and Path("/etc/nv_tegra_release").exists()
    except Exception:
        return False

def _jetson_cuda_env():
    """
    Compose an environment with common Jetson CUDA library paths so torch can find drivers.
    """
    env = os.environ.copy()
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
    ]
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for p in cuda_paths:
        if p not in ld_path:
            ld_path = f"{ld_path}:{p}" if ld_path else p
    env["LD_LIBRARY_PATH"] = ld_path
    return env

def _jetson_release_info():
    try:
        return Path("/etc/nv_tegra_release").read_text()
    except Exception:
        return None

def _jetson_torch_spec():
    """
    Return (extra_index_url, torch_wheel_url, torch_version, torchvision_version) for Jetson wheels.
    Environment overrides:
      JETSON_PYTORCH_INDEX (base URL, e.g., https://developer.download.nvidia.com/compute/redist/jp/v60)
      JETSON_TORCH_WHEEL (full wheel URL)
      JETSON_TORCH_VERSION
      JETSON_TORCHVISION_VERSION
    """
    env_index = os.environ.get("JETSON_PYTORCH_INDEX")
    env_wheel = os.environ.get("JETSON_TORCH_WHEEL")
    env_torch = os.environ.get("JETSON_TORCH_VERSION")
    env_vision = os.environ.get("JETSON_TORCHVISION_VERSION")

    release_info = _jetson_release_info() or ""

    # Candidate indexes (env override > derived > common fallbacks)
    candidates = []
    if env_index:
        candidates.append(env_index.rstrip("/"))
    else:
        if "R36" in release_info or "JP6" in release_info:
            candidates.append("https://developer.download.nvidia.com/compute/redist/jp/v60")
        if "R35" in release_info or "JP5" in release_info:
            candidates.append("https://developer.download.nvidia.com/compute/redist/jp/v51")
    # Always add common fallbacks if nothing derived
    if not candidates:
        candidates.extend(
            [
                "https://developer.download.nvidia.com/compute/redist/jp/v60",
                "https://developer.download.nvidia.com/compute/redist/jp/v51",
            ]
        )

    index_url = None
    wheel_url = None
    torch_ver = env_torch
    vision_ver = env_vision
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # If a wheel URL is provided explicitly, use it
    if env_wheel:
        wheel_url = env_wheel
        # Try to infer torch version from wheel name
        m = re.search(r"torch-([0-9a-zA-Z\\.\\+]+).*\\.whl", wheel_url)
        torch_ver = env_torch or (m.group(1).split("+")[0] if m else None)
        return index_url, wheel_url, torch_ver, vision_ver

    # Discover the latest torch wheel from the NVIDIA index (try candidates in order)
    for idx in candidates:
        try:
            with urllib.request.urlopen(f"{idx}/pytorch/") as resp:
                html = resp.read().decode("utf-8", "ignore")
            pattern = rf'href="(torch-([0-9a-zA-Z\\.\\+]+)\\.nv[^"]*{py_tag}[^"]*linux_aarch64\\.whl)"'
            matches = re.findall(pattern, html)
            best = None
            for fname, ver in matches:
                base_ver = ver.split("+")[0]
                try:
                    ver_obj = Version(base_ver)
                except Exception:
                    continue
                if best is None or ver_obj > best[0]:
                    best = (ver_obj, fname, base_ver)
            if best:
                index_url = idx
                wheel_url = f"{idx}/pytorch/{best[1]}"
                torch_ver = torch_ver or best[2]
                break
        except Exception as e:
            print(f"⚠️  Could not scrape Jetson torch wheels from {idx}: {e}")

    # Fallback to known wheel URLs if scraping failed
    if wheel_url is None:
        # Map of known Jetson PyTorch wheels by Python version and JetPack version
        fallback_wheels = {
            "cp312": [
                # JP6 (v60) - CP312 wheels (check NVIDIA site for availability)
                ("https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp312-cp312-linux_aarch64.whl", "2.4.0", "0.19.0a0"),
            ],
            "cp311": [
                # JP6 (v60) - CP311 wheels
                ("https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp311-cp311-linux_aarch64.whl", "2.4.0", "0.19.0a0"),
            ],
            "cp310": [
                # JP6 (v60) - CP310 wheels
                ("https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl", "2.4.0", "0.19.0a0"),
                ("https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+f70bd71a48.nv24.06.15634931-cp310-cp310-linux_aarch64.whl", "2.4.0", "0.19.0a0"),
            ],
            "cp38": [
                # JP5 (v51) - CP38 wheels
                ("https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl", "2.0.0", "0.15.2"),
            ],
        }

        # Try to find a wheel for the current Python version
        if py_tag in fallback_wheels:
            for url, fb_torch_ver, fb_vision_ver in fallback_wheels[py_tag]:
                wheel_url = url
                torch_ver = torch_ver or fb_torch_ver
                vision_ver = vision_ver or fb_vision_ver
                index_url = url.rsplit("/pytorch/", 1)[0]
                print(f"ℹ️  Using fallback Jetson torch wheel: {wheel_url}")
                print(f"   Fallback torch version: {torch_ver}, torchvision version: {vision_ver}")
                break
        else:
            print(f"⚠️  No fallback wheel available for Python {sys.version_info.major}.{sys.version_info.minor} ({py_tag})")
            print(f"   Available fallback wheels: {', '.join(fallback_wheels.keys())}")
            print(f"   You may need to:")
            print(f"   1. Use a compatible Python version (3.8, 3.10, 3.11)")
            print(f"   2. Set JETSON_TORCH_WHEEL to a compatible wheel URL")
            print(f"   3. Build PyTorch from source")

    # Map torch version to a torchvision version (source build) as best-effort (only if not already set)
    if vision_ver is None and torch_ver:
        major_minor = ".".join(torch_ver.split(".")[:2])
        # torch 2.4.x -> vision 0.19.x; torch 2.3.x -> vision 0.18.x; torch 2.1.x -> vision 0.15.x
        vision_map = {
            "2.4": "0.19.0a0",
            "2.3": "0.18.0",
            "2.2": "0.17.0",
            "2.1": "0.15.2",
            "2.0": "0.15.2",
        }
        vision_ver = vision_map.get(major_minor, "0.18.0")

    return index_url, wheel_url, torch_ver, vision_ver

def bootstrap_environment():
    """Bootstrap virtual environment and install dependencies."""
    base_dir = Path(__file__).parent.absolute()
    venv_dir = base_dir / "venv"
    is_jetson = _is_jetson()

    print("=" * 70)
    print("🚀 Depth Anything 3 - API Server Bootstrap")
    print("=" * 70)
    if is_jetson:
        print("ℹ️  Detected Jetson/aarch64 device. Using Jetson-friendly install path.")

    # Check if already in venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Already running in virtual environment")
        return True

    # On Jetson, check Python version compatibility
    if is_jetson:
        py_major = sys.version_info.major
        py_minor = sys.version_info.minor
        current_py_tag = f"cp{py_major}{py_minor}"

        # JetPack 6.0 typically only has cp310 wheels
        supported_versions = ["cp310"]

        if current_py_tag not in supported_versions:
            print(f"\n⚠️  WARNING: Python {py_major}.{py_minor} may not have pre-built PyTorch wheels for Jetson")
            print(f"   Current Python: {sys.executable}")
            print(f"   Supported versions: {', '.join([v.replace('cp', 'Python 3.') for v in supported_versions])}")

            # Try to find Python 3.10
            for py_cmd in ["python3.10", "python3.10-venv"]:
                try:
                    result = subprocess.run([py_cmd, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"\n✓ Found compatible Python: {py_cmd}")
                        print(f"   Please run this script with: {py_cmd} {' '.join(sys.argv)}")
                        print(f"   Or create a venv manually: {py_cmd} -m venv venv && ./venv/bin/python3 {sys.argv[0]}")
                        return False
                except FileNotFoundError:
                    continue

            print(f"\n❌ Python 3.10 not found on system.")
            print(f"   Install it with: sudo apt install python3.10 python3.10-venv")
            print(f"   Then run: python3.10 {sys.argv[0]}")
            return False

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
    reinstall_needed = not marker_file.exists()

    def _mark_reinstall(reason: str):
        nonlocal reinstall_needed
        reinstall_needed = True
        print(reason)
        try:
            marker_file.unlink()
        except FileNotFoundError:
            pass

    # If we previously installed deps but new requirements were added (e.g., addict),
    # verify the venv still has the critical modules. If not, force a reinstall.
    if not reinstall_needed and requirements_file.exists():
        critical_modules = ["addict", "omegaconf", "einops"]
        check_code = (
            "import importlib.util, sys; "
            f"mods = {critical_modules!r}; "
            "missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
            "print(','.join(missing)); "
            "sys.exit(1 if missing else 0)"
        )
        result = subprocess.run(
            [str(venv_python), "-c", check_code],
            capture_output=True,
            text=True,
        )
        missing = [m for m in result.stdout.strip().split(",") if m]
        if result.returncode != 0 and missing:
            _mark_reinstall(f"\n⚠️ Detected missing packages in venv: {', '.join(missing)}\n   Reinstalling requirements to pick up new dependencies...")

    # On Jetson, strip incompatible xformers even if we don't reinstall (aarch64 wheels unavailable)
    if is_jetson:
        subprocess.run(
            [str(venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "pip"), "uninstall", "-y", "xformers"],
            check=False,
            env=_jetson_cuda_env(),
        )

    # Enforce numpy < 2 (depth-anything-3 requires it)
    if not reinstall_needed:
        numpy_check = (
            "import sys\n"
            "try:\n"
            " import numpy as np\n"
            " major = int(np.__version__.split('.')[0])\n"
            " sys.exit(0 if major < 2 else 1)\n"
            "except Exception as e:\n"
            " sys.exit(2)\n"
        )
        np_res = subprocess.run(
            [str(venv_python), "-c", numpy_check],
            capture_output=True,
            text=True,
            env=_jetson_cuda_env() if is_jetson else None,
        )
        if np_res.returncode != 0:
            _mark_reinstall("\n⚠️ Detected incompatible numpy (>=2). Will reinstall with numpy<2.")

    # On Jetson, ensure the installed torch actually reports CUDA available; if not, force reinstall
    if not reinstall_needed and is_jetson:
        torch_check = (
            "import sys\n"
            "try:\n"
            " import torch\n"
            " ok = torch.cuda.is_available()\n"
            " print('TORCH_VERSION', torch.__version__)\n"
            " print('CUDA_AVAILABLE', ok)\n"
            " sys.exit(0 if ok else 1)\n"
            "except Exception as e:\n"
            " print('ERR', e)\n"
            " sys.exit(2)\n"
        )
        tc_res = subprocess.run(
            [str(venv_python), "-c", torch_check],
            capture_output=True,
            text=True,
        )
        if tc_res.returncode != 0:
            output = tc_res.stdout.strip() or tc_res.stderr.strip()
            _mark_reinstall(
                "\n⚠️ Detected a Jetson without a CUDA-enabled torch in the venv."
                f"\n   torch check output: {output or 'n/a'}"
                "\n   Will reinstall torch from the Jetson wheel index."
            )

    if reinstall_needed and requirements_file.exists():
        print("\n📥 Installing dependencies from requirements.txt...")
        print("⏳ This may take several minutes...")

        try:
            # Upgrade pip first
            print("\n📦 Upgrading pip...")
            subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)

            # Always upgrade packaging tools; helps resolve correct wheels on Jetson
            subprocess.run([str(venv_pip), "install", "--upgrade", "setuptools", "wheel"], check=True)

            # Remove xformers on Jetson (no aarch64 wheels and may conflict with torch version)
            if is_jetson:
                subprocess.run([str(venv_pip), "uninstall", "-y", "xformers"], check=False)

            # Install torch first (required by xformers and other deps)
            print("\n📦 Installing PyTorch (this is a large package)...")
            if is_jetson:
                jp_index, wheel_url, torch_ver, vision_ver = _jetson_torch_spec()
                if wheel_url:
                    if jp_index:
                        print(f"   Using Jetson wheel index: {jp_index}")
                    print(f"   Installing torch from wheel: {wheel_url}")
                    if torch_ver:
                        print(f"   Torch version: {torch_ver}")
                    if vision_ver:
                        print(f"   Torchvision version: {vision_ver}")
                    try:
                        subprocess.run(
                            [str(venv_pip), "install", wheel_url],
                            check=True,
                            env=_jetson_cuda_env(),
                        )
                        print("✓ PyTorch installed successfully")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Jetson PyTorch install failed with error: {e}")
                        print("   Set JETSON_TORCH_WHEEL to a wheel URL from https://developer.download.nvidia.com/compute/redist/jp/")
                        print("   or install torch/torchvision manually inside the venv.")
                        raise
                else:
                    print("❌ Could not find a CUDA torch wheel for Jetson automatically.")
                    print("   Set JETSON_TORCH_WHEEL to a wheel URL from https://developer.download.nvidia.com/compute/redist/jp/")
                    print("   or install torch/torchvision manually inside the venv.")
                    raise SystemExit(1)
            else:
                subprocess.run([str(venv_pip), "install", "torch>=2", "torchvision"], check=True)

            # Ensure numpy is pinned < 2 after torch install
            subprocess.run([str(venv_pip), "install", "--upgrade", "numpy<2"], check=True)

            # Install torchvision on Jetson (build from source - no aarch64 wheels available)
            if is_jetson:
                # First, uninstall any existing incompatible torchvision
                subprocess.run([str(venv_pip), "uninstall", "-y", "torchvision"], check=False, capture_output=True)

                # Determine the correct torchvision branch/tag for the torch version
                torch_major_minor = torch_ver.split(".")[:2] if torch_ver else ["2", "4"]
                vision_branch_map = {
                    ("2", "4"): "v0.19.0",
                    ("2", "3"): "v0.18.0",
                    ("2", "2"): "v0.17.0",
                    ("2", "1"): "v0.16.0",
                    ("2", "0"): "v0.15.2",
                }
                vision_tag = vision_branch_map.get(tuple(torch_major_minor), "v0.19.0")

                print(f"   Installing torchvision from source (tag {vision_tag}, compatible with torch {torch_ver})")
                print(f"   ⚠️  This may take 15-30 minutes on Jetson. Please be patient...")

                try:
                    # Install torchvision from GitHub source at the correct tag
                    subprocess.run(
                        [
                            str(venv_pip),
                            "install",
                            f"git+https://github.com/pytorch/vision.git@{vision_tag}",
                        ],
                        check=True,
                        env=_jetson_cuda_env(),
                        timeout=3600,  # 1 hour timeout for building
                    )
                    print("✓ Torchvision installed successfully from source")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Torchvision build from source failed: {e}")
                    print("   Trying to install latest compatible version from PyPI...")
                    # Fallback: try to find any compatible torchvision on PyPI
                    try:
                        subprocess.run(
                            [str(venv_pip), "install", "torchvision", "--upgrade"],
                            check=True,
                            env=_jetson_cuda_env(),
                        )
                    except subprocess.CalledProcessError:
                        print("❌ Could not install torchvision. You may need to build it manually.")
                        print("   See: https://github.com/pytorch/vision#installation")
                except subprocess.TimeoutExpired:
                    print("⚠️  Torchvision build timed out after 1 hour")
                    print("   You may need to build it manually with more resources")

            # Verify CUDA availability on Jetson immediately after torch install
            if is_jetson:
                cuda_probe = (
                    "import torch, sys; "
                    "ok = torch.cuda.is_available(); "
                    "print('CUDA_AVAILABLE', ok); "
                    "sys.exit(0 if ok else 1)"
                )
                probe_res = subprocess.run(
                    [str(venv_python), "-c", cuda_probe],
                    capture_output=True,
                    text=True,
                    env=_jetson_cuda_env(),
                )
                if probe_res.returncode != 0:
                    print("❌ CUDA not available after installing Jetson torch.")
                    print("   This likely means the installed wheel is CPU-only or CUDA drivers are not visible.")
                    print("   If you overrode versions, set JETSON_TORCH_WHEEL to a CUDA-enabled wheel URL.")
                    print("   Otherwise, check JetPack/CUDA installation on the device.")
                    print(f"   Probe output: {probe_res.stdout.strip() or probe_res.stderr.strip()}")
                    raise SystemExit(1)

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
            if is_jetson:
                print("\nℹ️  Skipping xformers on Jetson (no aarch64 wheels available).")
            else:
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

# After bootstrap: import remaining dependencies
import socket
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import torch
from PIL import Image
import trimesh

# Provide a safe shim for older torch versions that lack flash attention detection (e.g., Jetson wheels)
if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
    def _no_flash_attention_available():
        return False
    torch.backends.cuda.is_flash_attention_available = _no_flash_attention_available

# Enable TF32 where available (Ampere+ → tensor cores) to maximize throughput
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.registry import MODEL_REGISTRY
except ImportError as e:
    print(f"❌ Error importing DA3 modules: {e}")
    raise

# Optional: extended API helpers (e.g., /api/v1/process/image)
try:
    from api_endpoints import create_api_routes
except ImportError:
    create_api_routes = None

# ============================================================================
# MODEL REGISTRY WITH METADATA
# ============================================================================

AVAILABLE_MODELS = {
    key: {
        "name": data["name"],
        "hf_path": data["hf_path"],
        "description": data.get("description", ""),
        "size": data.get("size", "unknown"),
        "speed": data.get("speed", "unknown"),
        "quality": data.get("quality", "unknown"),
        "recommended_for": data.get("recommended_for", "General use")
    }
    for key, data in {
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
    }.items()
}

# ============================================================================
# APP SETUP
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

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# ============================================================================
# STATE
# ============================================================================

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

# Register optional extended API routes
if create_api_routes:
    try:
        create_api_routes(app, state)
        print("✓ Extended API routes registered")
    except Exception as e:
        print(f"⚠️  Warning: Could not register extended API routes: {e}")

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

    percentile_5 = np.percentile(vertices[:, 1], 5)  # Bottom 5% in Y
    floor_candidates = vertices[vertices[:, 1] < percentile_5 + 0.5]

    if len(floor_candidates) < 100:
        print("⚠️ Not enough floor candidates, skipping alignment")
        return vertices, np.eye(4)

    best_plane = None
    best_inliers = 0

    for _ in range(50):  # 50 iterations
        sample_idx = np.random.choice(len(floor_candidates), 3, replace=False)
        sample_points = floor_candidates[sample_idx]

        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        normal_len = np.linalg.norm(normal)

        if normal_len < 1e-6:
            continue

        normal = normal / normal_len
        d = -np.dot(normal, sample_points[0])

        distances = np.abs(floor_candidates @ normal + d)
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

    target_normal = np.array([0, 1, 0])
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis_len = np.linalg.norm(rotation_axis)

    if rotation_axis_len < 1e-6:
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = rotation_axis / rotation_axis_len
        angle = np.arccos(np.clip(np.dot(normal, target_normal), -1, 1))
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    aligned_vertices = vertices @ rotation_matrix.T
    floor_y = np.percentile(aligned_vertices[:, 1], 5)
    translation = np.array([0, -floor_y, 0])
    aligned_vertices = aligned_vertices + translation

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = rotation_matrix @ translation

    print(f"✓ Floor aligned to y=0 (translated by {floor_y:.3f})")

    return aligned_vertices.tolist(), transform_matrix.tolist()

# ============================================================================
# ROUTES
# ============================================================================

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
    thread.daemon = True
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
                colors = np.vstack(colors) if colors else np.ones_like(vertices) * 255

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

    if not state.model_ready:
        return jsonify({"error": "Model not loaded"}), 503

    def parse_bool(key, default=False):
        val = request.form.get(key)
        if val is None:
            return default
        return str(val).lower() in {"1", "true", "yes", "on"}

    resolution = int(request.form.get('resolution', 504))
    max_points = int(request.form.get('max_points', 1000000))
    process_res_method = request.form.get('process_res_method', 'upper_bound_resize')
    valid_methods = {'upper_bound_resize', 'upper_bound_crop', 'lower_bound_resize', 'lower_bound_crop'}
    if process_res_method not in valid_methods:
        process_res_method = 'upper_bound_resize'

    align_to_input_scale = parse_bool('align_to_input_ext_scale', True)
    infer_gs = parse_bool('infer_gs', False)
    conf_thresh_percentile = float(request.form.get('conf_thresh_percentile', 40.0))
    show_cameras = parse_bool('show_cameras', True)
    feat_vis_fps = int(request.form.get('feat_vis_fps', 15))
    include_confidence = parse_bool('include_confidence', False)
    apply_confidence_filter = parse_bool('apply_confidence_filter', False)

    export_feat_layers_raw = request.form.get('export_feat_layers', '')
    export_feat_layers = []
    if export_feat_layers_raw:
        for layer in export_feat_layers_raw.split(','):
            layer = layer.strip()
            if layer.isdigit():
                export_feat_layers.append(int(layer))

    job_id = f"job_{int(time.time() * 1000)}"

    def process_async():
        try:
            file_ext = filepath.suffix.lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

            if is_video:
                from moviepy.editor import VideoFileClip

                clip = VideoFileClip(str(filepath))
                fps = clip.fps
                sample_rate = max(1, int(fps * 0.5))
                frames = []

                for i, frame in enumerate(clip.iter_frames()):
                    if i % sample_rate == 0:
                        frames.append(Image.fromarray(frame))
                    if len(frames) >= 20:
                        break

                clip.close()

                prediction = state.model.inference(
                    image=frames,
                    process_res=resolution,
                    process_res_method=process_res_method,
                    num_max_points=max_points,
                    align_to_input_ext_scale=align_to_input_scale,
                    infer_gs=infer_gs,
                    export_feat_layers=export_feat_layers,
                    conf_thresh_percentile=conf_thresh_percentile,
                    show_cameras=show_cameras,
                    feat_vis_fps=feat_vis_fps
                )
            else:
                img = Image.open(filepath).convert('RGB')

                prediction = state.model.inference(
                    image=[img],
                    process_res=resolution,
                    process_res_method=process_res_method,
                    num_max_points=max_points,
                    align_to_input_ext_scale=align_to_input_scale,
                    infer_gs=infer_gs,
                    export_feat_layers=export_feat_layers,
                    conf_thresh_percentile=conf_thresh_percentile,
                    show_cameras=show_cameras,
                    feat_vis_fps=feat_vis_fps
                )

            try:
                from api_endpoints import prediction_to_point_cloud_json
                point_cloud = prediction_to_point_cloud_json(
                    prediction,
                    max_points=max_points,
                    include_confidence=include_confidence or apply_confidence_filter
                )

                if apply_confidence_filter and "confidence" in point_cloud:
                    conf_values = np.array(point_cloud["confidence"], dtype=np.float32)
                    vertices = np.array(point_cloud["vertices"], dtype=np.float32)
                    colors = np.array(point_cloud["colors"], dtype=np.float32)

                    threshold = np.percentile(conf_values, conf_thresh_percentile)
                    mask = conf_values >= threshold

                    vertices = vertices[mask]
                    colors = colors[mask]
                    conf_values = conf_values[mask]

                    point_cloud["vertices"] = vertices.tolist()
                    point_cloud["colors"] = colors.tolist()
                    point_cloud["metadata"]["num_points"] = int(len(vertices))
                    if include_confidence:
                        point_cloud["confidence"] = conf_values.tolist()

                state.current_pointcloud = {
                    "vertices": point_cloud["vertices"],
                    "colors": point_cloud["colors"],
                    "metadata": {
                        "num_points": point_cloud["metadata"]["num_points"],
                        "resolution": resolution,
                        "filename": filename
                    }
                }
                if include_confidence and "confidence" in point_cloud:
                    state.current_pointcloud["confidence"] = point_cloud["confidence"]
            except ImportError:
                points_list = []
                colors_list = []
                conf_list = []

                confidence_maps = None
                if apply_confidence_filter or include_confidence:
                    confidence_maps = getattr(prediction, "conf", None)

                for i in range(len(prediction.depth)):
                    depth = prediction.depth[i]
                    image = prediction.processed_images[i]
                    conf_map = None
                    if confidence_maps is not None and i < len(confidence_maps):
                        conf_map = confidence_maps[i]

                    ixt = prediction.intrinsics[i]
                    fx, fy = ixt[0, 0], ixt[1, 1]
                    cx, cy = ixt[0, 2], ixt[1, 2]

                    h, w = depth.shape
                    y_coords, x_coords = np.mgrid[0:h, 0:w]

                    x3d = -(x_coords - cx) * depth / fx
                    y3d = -(y_coords - cy) * depth / fy
                    z3d = depth

                    points = np.stack([x3d, y3d, z3d], axis=-1).reshape(-1, 3)
                    colors = image.reshape(-1, 3)
                    conf_flat = conf_map.reshape(-1) if conf_map is not None else None

                    valid = depth.reshape(-1) > 0
                    points = points[valid]
                    colors = colors[valid]
                    if conf_flat is not None:
                        conf_flat = conf_flat[valid]

                    points_list.append(points)
                    colors_list.append(colors)
                    if conf_flat is not None:
                        conf_list.append(conf_flat)

                all_points = np.vstack(points_list)
                all_colors = np.vstack(colors_list)
                all_conf = np.hstack(conf_list) if conf_list else None

                if apply_confidence_filter and all_conf is not None and len(all_conf) > 0:
                    threshold = np.percentile(all_conf, conf_thresh_percentile)
                    mask = all_conf >= threshold
                    all_points = all_points[mask]
                    all_colors = all_colors[mask]
                    all_conf = all_conf[mask]

                if max_points and len(all_points) > max_points:
                    indices = np.random.choice(len(all_points), max_points, replace=False)
                    all_points = all_points[indices]
                    all_colors = all_colors[indices]
                    if all_conf is not None:
                        all_conf = all_conf[indices]

                state.current_pointcloud = {
                    "vertices": all_points.tolist(),
                    "colors": all_colors.tolist(),
                    "metadata": {
                        "num_points": len(all_points),
                        "resolution": resolution,
                        "filename": filename
                    }
                }
                if include_confidence and all_conf is not None:
                    state.current_pointcloud["confidence"] = all_conf.tolist()

            if state.current_pointcloud and "metadata" in state.current_pointcloud:
                state.current_pointcloud["metadata"]["config"] = {
                    "process_res": resolution,
                    "process_res_method": process_res_method,
                    "max_points": max_points,
                    "align_to_input_ext_scale": align_to_input_scale,
                    "infer_gs": infer_gs,
                    "export_feat_layers": export_feat_layers,
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "show_cameras": show_cameras,
                    "feat_vis_fps": feat_vis_fps,
                    "include_confidence": include_confidence,
                    "apply_confidence_filter": apply_confidence_filter
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
    thread.daemon = True
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

    aligned_vertices, transform = detect_and_align_floor(vertices, colors)

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

        if colors.max() <= 1.0:
            colors = (colors * 255.0).clip(0, 255)
        colors = colors.astype(np.uint8)

        src_name = state.current_pointcloud.get("metadata", {}).get("filename", "point_cloud")
        glb_name = f"{Path(src_name).stem}.glb"

        output_dir = Path(app.config['OUTPUT_FOLDER'])
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / glb_name

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
# MAIN ENTRY POINT
# ============================================================================

def setup_signal_handlers():
    """Handle termination signals so Ctrl+C/Z exit cleanly."""
    import signal
    def _graceful_exit(signum, frame):
        print("\n🛑 Received termination signal, shutting down gracefully...")
        raise SystemExit(0)

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGTSTP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _graceful_exit)
        except Exception:
            pass

if __name__ == '__main__':
    setup_signal_handlers()
    print("\n" + "="*70)
    print("🚀 Depth Anything 3 - API Server (no UI)")
    print("="*70)
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print("="*70)

    port = find_available_port()
    print(f"\n📡 Starting API server on http://0.0.0.0:{port}")
    print("Endpoints:")
    print("  • /api/models/list")
    print("  • /api/models/select")
    print("  • /api/load_model")
    print("  • /api/process")
    print("  • /api/job/<job_id>")
    print("  • /api/floor_align")
    print("  • /api/export/glb")
    print("  • /api/v1/* aliases")
    print("\n⚠️  Press Ctrl+C to stop the server")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
