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
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda-12.4/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
        "/usr/lib/aarch64-linux-gnu/tegra-egl",
    ]
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for p in cuda_paths:
        if Path(p).exists() and p not in ld_path:
            ld_path = f"{ld_path}:{p}" if ld_path else p
    env["LD_LIBRARY_PATH"] = ld_path

    # Also set CUDA_HOME if not set
    if "CUDA_HOME" not in env:
        for cuda_home in ["/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.2", "/usr/local/cuda-12.4", "/usr/local/cuda-12.6"]:
            if Path(cuda_home).exists():
                env["CUDA_HOME"] = cuda_home
                break

    return env

def _install_jetson_cusparselt():
    """
    Install cuSPARSELt library required by PyTorch 2.4+ on JetPack 6.0+.
    This library is not included by default in JetPack but is required for PyTorch.
    Returns True if installed successfully or already present, False otherwise.
    """
    # Check if already installed
    lib_paths = [
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.2/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.4/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.6/lib64/libcusparseLt.so.0",
        "/usr/lib/libcusparseLt.so.0",
    ]

    for lib_path in lib_paths:
        if Path(lib_path).exists():
            print(f"✓ libcusparseLt.so.0 found at {lib_path}")
            return True

    print("⚠️  libcusparseLt.so.0 not found - required by PyTorch 2.4+")
    print("   Installing cuSPARSELt library...")

    # Find CUDA version from nvcc
    cuda_version = None
    try:
        nvcc_output = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse version from output like "release 12.2, V12.2.140"
        import re
        match = re.search(r"release (\d+\.\d+)", nvcc_output.stdout)
        if match:
            cuda_version = match.group(1)
            print(f"   Detected CUDA version: {cuda_version}")
    except Exception as e:
        print(f"   Could not detect CUDA version: {e}")

    # Find CUDA home
    cuda_home = None
    for candidate in ["/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.2", "/usr/local/cuda-12.4", "/usr/local/cuda-12.6"]:
        if Path(candidate).exists():
            cuda_home = candidate
            break

    if not cuda_home:
        print("❌ CUDA installation not found. Cannot install cuSPARSELt.")
        return False

    try:
        import tempfile
        import tarfile
        import urllib.request

        # Determine cuSPARSELt version based on CUDA version
        # Following PyTorch's official installation script:
        # https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cusparselt.sh
        if cuda_version and cuda_version.startswith("12."):
            # CUDA 12.1-12.6 -> cuSPARSELt 0.5.2.1 (PyTorch's recommended version)
            CUSPARSELT_VERSION = "0.5.2.1"
            arch = "sbsa"  # ARM architecture (Jetson uses ARM/aarch64)
        elif cuda_version and cuda_version.startswith("11.8"):
            # CUDA 11.8 -> cuSPARSELt 0.4.0.7
            CUSPARSELT_VERSION = "0.4.0.7"
            arch = "x86_64"
        else:
            # Default to version compatible with CUDA 12.x
            print(f"   Unknown CUDA version {cuda_version}, using default cuSPARSELt 0.5.2.1")
            CUSPARSELT_VERSION = "0.5.2.1"
            arch = "sbsa"

        CUSPARSELT_NAME = f"libcusparse_lt-linux-{arch}-{CUSPARSELT_VERSION}-archive"
        CUSPARSELT_URL = f"https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-{arch}/{CUSPARSELT_NAME}.tar.xz"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            archive_path = tmpdir / f"{CUSPARSELT_NAME}.tar.xz"

            print(f"   Downloading cuSPARSELt {CUSPARSELT_VERSION} for {arch}...")
            print(f"   URL: {CUSPARSELT_URL}")

            # Retry logic like PyTorch's script
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(CUSPARSELT_URL, archive_path)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"   Download attempt {attempt + 1} failed, retrying...")

            print("   Extracting archive...")
            with tarfile.open(archive_path, 'r:xz') as tar:
                tar.extractall(tmpdir)

            # Find the extracted directory
            extracted_dir = tmpdir / CUSPARSELT_NAME

            if not extracted_dir.exists():
                print(f"❌ Expected directory not found: {extracted_dir}")
                return False

            # Copy libraries and headers
            print(f"   Installing to {cuda_home}...")

            # Copy include files (use -a to preserve symlinks like PyTorch's script)
            include_src = extracted_dir / "include"
            include_dst = Path(cuda_home) / "include"
            if include_src.exists():
                subprocess.run(["sudo", "cp", "-a", str(include_src) + "/.", str(include_dst)], check=True)

            # Copy library files (use -a to preserve symlinks like PyTorch's script)
            lib_src = extracted_dir / "lib"
            lib_dst = Path(cuda_home) / "lib64"
            if lib_src.exists():
                subprocess.run(["sudo", "cp", "-a", str(lib_src) + "/.", str(lib_dst)], check=True)

            # Update linker cache
            print("   Updating linker cache...")
            subprocess.run(["sudo", "ldconfig"], check=True)

            print("✓ cuSPARSELt installed successfully")
            return True

    except Exception as e:
        print(f"❌ Failed to install cuSPARSELt: {e}")
        print("\n   MANUAL INSTALLATION:")
        print("   You can install cuSPARSELt manually using one of these methods:")
        print("\n   Method 1 - PyTorch's official script (recommended):")
        print("   wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh")
        print("   export CUDA_VERSION=12.2  # or your CUDA version")
        print("   sudo bash ./install_cusparselt.sh")
        print("\n   Method 2 - Debian package:")
        print("   wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb")
        print("   sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb")
        print("   sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/")
        print("   sudo apt-get update")
        print("   sudo apt-get -y install libcusparselt0 libcusparselt-dev")
        print("\n   Method 3 - Download from:")
        print("   https://developer.nvidia.com/cusparselt-downloads")
        return False

def _jetson_release_info():
    try:
        return Path("/etc/nv_tegra_release").read_text()
    except Exception:
        return None

def _jetson_system_diagnostics():
    """
    Comprehensive Jetson system diagnostics similar to the bash probe command.
    Returns a dict with system information for version matching and debugging.
    """
    info = {
        "is_jetson": False,
        "jetson_module": None,
        "jetpack_version": None,
        "cuda_version": None,
        "cuda_arch": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": False,
        "torchvision_version": None,
        "torchvision_nms_available": False,
    }

    # Basic Jetson detection
    info["is_jetson"] = _is_jetson()
    if not info["is_jetson"]:
        return info

    # Parse Jetson release info
    release_text = _jetson_release_info() or ""
    if release_text:
        # Extract revision (e.g., R36 -> JP6, R35 -> JP5)
        rev_match = re.search(r"R(\d+)", release_text)
        if rev_match:
            r_version = int(rev_match.group(1))
            # Map R36 -> JP6, R35 -> JP5, etc.
            jp_version = r_version - 30  # R36->6, R35->5, R34->4
            info["jetpack_version"] = f"JP{jp_version}.x"

    # Query dpkg for nvidia packages
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "nvidia-l4t-core", "nvidia-jetpack"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "nvidia-l4t-core" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        info["l4t_version"] = parts[1]
    except Exception:
        pass

    # Get CUDA version from nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            info["cuda_version"] = match.group(1)
    except Exception:
        pass

    # Get CUDA architecture from environment
    info["cuda_arch"] = os.environ.get("JETSON_CUDA_ARCH_BIN", "8.7")

    # Try to get Jetson module name
    try:
        result = subprocess.run(
            ["cat", "/proc/device-tree/model"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            info["jetson_module"] = result.stdout.strip().rstrip("\x00")
    except Exception:
        pass

    return info

def _probe_torch_torchvision(venv_python, cuda_env=None):
    """
    Probe the installed torch/torchvision to check versions and NMS availability.
    Returns a dict with the probe results.
    """
    probe_code = """
import sys
import json
result = {
    "torch_version": None,
    "torch_cuda_version": None,
    "torch_cuda_available": False,
    "torchvision_version": None,
    "torchvision_nms_available": False,
    "error": None
}

try:
    import torch
    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    result["torch_cuda_available"] = torch.cuda.is_available()
except Exception as e:
    result["error"] = f"torch import: {repr(e)}"

try:
    import torchvision
    result["torchvision_version"] = torchvision.__version__
    # Check if nms operator exists
    import torchvision.ops
    result["torchvision_nms_available"] = hasattr(torch.ops.torchvision, "nms")
except Exception as e:
    if result["error"]:
        result["error"] += f"; torchvision: {repr(e)}"
    else:
        result["error"] = f"torchvision: {repr(e)}"

print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            [str(venv_python), "-c", probe_code],
            capture_output=True,
            text=True,
            env=cuda_env,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            return {"error": f"probe failed: {result.stderr}"}
    except Exception as e:
        return {"error": f"probe exception: {repr(e)}"}

def _rebuild_torchvision_jetson(venv_python, venv_pip, torch_version, cuda_env):
    """
    Rebuild torchvision from source with CUDA extensions on Jetson.
    Returns True if successful, False otherwise.
    """
    print("\n🔨 Rebuilding torchvision from source with CUDA extensions...")
    print("   This may take 15-30 minutes on Jetson. Please be patient...")

    # Determine the correct torchvision tag for the torch version
    torch_parts = torch_version.split(".")
    torch_major_minor = tuple(torch_parts[:2]) if len(torch_parts) >= 2 else ("2", "4")

    vision_branch_map = {
        ("2", "5"): "v0.20.0",
        ("2", "4"): "v0.19.0",
        ("2", "3"): "v0.18.0",
        ("2", "2"): "v0.17.0",
        ("2", "1"): "v0.16.0",
        ("2", "0"): "v0.15.2",
    }
    vision_tag = vision_branch_map.get(torch_major_minor, "v0.19.0")

    print(f"   Torch version: {torch_version}")
    print(f"   Target torchvision tag: {vision_tag}")

    # Get system diagnostics for CUDA arch
    diag = _jetson_system_diagnostics()
    cuda_arch = diag.get("cuda_arch", "8.7")

    # Find CUDA home
    cuda_home = None
    for candidate in ["/usr/local/cuda-12.2", "/usr/local/cuda-12", "/usr/local/cuda"]:
        if Path(candidate).exists():
            cuda_home = candidate
            break

    if not cuda_home:
        print("❌ CUDA installation not found")
        return False

    print(f"   Using CUDA_HOME: {cuda_home}")
    print(f"   Using TORCH_CUDA_ARCH_LIST: {cuda_arch}")

    # Install system dependencies
    print("   Installing system build dependencies...")
    subprocess.run(
        [
            "sudo", "apt-get", "install", "-y",
            "libjpeg-dev", "zlib1g-dev", "libpython3-dev",
            "libopenblas-dev", "libavcodec-dev", "libavformat-dev", "libswscale-dev"
        ],
        check=False,
        capture_output=True
    )

    # Install Python build dependencies (including numpy for build)
    print("   Installing Python build tools...")
    try:
        subprocess.run(
            [str(venv_pip), "install", "--upgrade", "setuptools", "wheel", "ninja", "numpy"],
            check=True,
            capture_output=True,
            env=cuda_env
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to install build tools: {e}")
        return False

    # DON'T uninstall torchvision yet - only if build succeeds
    # This prevents leaving the system with no torchvision at all

    # Set up build environment with all necessary variables
    build_env = cuda_env.copy() if cuda_env else os.environ.copy()
    build_env.update({
        "CUDA_HOME": cuda_home,
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": cuda_arch,
        "MAX_JOBS": "4",  # Limit parallel jobs to avoid OOM
    })

    # Build and install torchvision
    print(f"   Building torchvision from {vision_tag}...")
    print("   (This will take 15-30 minutes - output will be shown)")

    try:
        # Use --no-build-isolation to ensure our CUDA_HOME is visible to the build
        result = subprocess.run(
            [
                str(venv_pip),
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",  # Critical: allows CUDA_HOME and FORCE_CUDA to be seen
                f"git+https://github.com/pytorch/vision.git@{vision_tag}",
            ],
            env=build_env,
            timeout=3600,  # 1 hour timeout
            capture_output=False,  # Show output for debugging
        )

        if result.returncode != 0:
            print("❌ Torchvision build failed")
            return False

        print("✓ Torchvision built and installed successfully")
        return True

    except subprocess.TimeoutExpired:
        print("❌ Torchvision build timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Torchvision build failed: {e}")
        return False

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

        # Display system diagnostics
        diag = _jetson_system_diagnostics()
        print("\n📊 Jetson System Diagnostics:")
        if diag.get("jetson_module"):
            print(f"   Module: {diag['jetson_module']}")
        if diag.get("jetpack_version"):
            print(f"   JetPack: {diag['jetpack_version']}")
        if diag.get("l4t_version"):
            print(f"   L4T: {diag['l4t_version']}")
        if diag.get("cuda_version"):
            print(f"   CUDA: {diag['cuda_version']}")
        if diag.get("cuda_arch"):
            print(f"   CUDA Arch: {diag['cuda_arch']}")
        print(f"   Python: {diag['python_version']}")
        print()

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

    # On Jetson, check if torchvision NMS operator is available; if not, force rebuild
    if not reinstall_needed and is_jetson:
        probe_result = _probe_torch_torchvision(venv_python, _jetson_cuda_env())
        if probe_result.get("error") or not probe_result.get("torchvision_nms_available"):
            _mark_reinstall(
                "\n⚠️ Detected torchvision without NMS operator (C++ extensions not compiled)."
                f"\n   Error: {probe_result.get('error', 'NMS not available')}"
                "\n   Will rebuild torchvision from source with CUDA extensions."
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
                # Install cuSPARSELt library if needed (required by PyTorch 2.4+)
                print("\n🔍 Checking for cuSPARSELt library...")
                _install_jetson_cusparselt()

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
                    # Install build dependencies for torchvision
                    print("   Installing build dependencies...")
                    subprocess.run(
                        [
                            "sudo", "apt-get", "install", "-y",
                            "libjpeg-dev", "zlib1g-dev", "libpython3-dev",
                            "libopenblas-dev", "libavcodec-dev", "libavformat-dev", "libswscale-dev"
                        ],
                        check=False,  # Don't fail if already installed
                        capture_output=True
                    )

                    # Install build tools (including numpy which is needed during build)
                    print("   Installing Python build tools...")
                    subprocess.run(
                        [str(venv_pip), "install", "--upgrade", "setuptools", "wheel", "ninja", "numpy"],
                        check=False,
                        capture_output=True,
                        env=_jetson_cuda_env()
                    )

                    # Find CUDA home for build environment
                    cuda_home = None
                    for candidate in ["/usr/local/cuda-12.2", "/usr/local/cuda-12", "/usr/local/cuda"]:
                        if Path(candidate).exists():
                            cuda_home = candidate
                            break

                    # Set up build environment
                    build_env = _jetson_cuda_env()
                    if cuda_home:
                        build_env["CUDA_HOME"] = cuda_home
                        build_env["FORCE_CUDA"] = "1"
                        build_env["TORCH_CUDA_ARCH_LIST"] = "8.7"
                        build_env["MAX_JOBS"] = "4"
                        print(f"   Using CUDA_HOME: {cuda_home}")

                    # Install torchvision from GitHub source at the correct tag
                    # Use --no-build-isolation to ensure CUDA_HOME is visible
                    subprocess.run(
                        [
                            str(venv_pip),
                            "install",
                            "--no-cache-dir",  # Force rebuild
                            "--force-reinstall",  # Force reinstall
                            "--no-deps",  # Keep the Jetson torch wheel; avoid pulling a CPU torch
                            "--no-build-isolation",  # Critical: allows CUDA_HOME to be seen
                            f"git+https://github.com/pytorch/vision.git@{vision_tag}",
                        ],
                        check=True,
                        env=build_env,
                        timeout=3600,  # 1 hour timeout for building
                    )
                    print("✓ Torchvision installed successfully from source")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Torchvision build from source failed: {e}")
                    print("   Trying to install latest compatible version from PyPI...")
                    # Fallback: try to find any compatible torchvision on PyPI
                    try:
                        subprocess.run(
                            [str(venv_pip), "install", "torchvision", "--upgrade", "--no-deps"],
                            check=True,
                            env=_jetson_cuda_env(),
                        )
                    except subprocess.CalledProcessError:
                        print("❌ Could not install torchvision. You may need to build it manually.")
                        print("   See: https://github.com/pytorch/vision#installation")
                except subprocess.TimeoutExpired:
                    print("⚠️  Torchvision build timed out after 1 hour")
                    print("   You may need to build it manually with more resources")

                # Verify torchvision imports correctly (check for NMS operator)
                print("   Verifying torchvision installation...")
                probe_result = _probe_torch_torchvision(venv_python, _jetson_cuda_env())

                if probe_result.get("error") or not probe_result.get("torchvision_nms_available"):
                    print(f"⚠️  Torchvision verification failed: {probe_result.get('error', 'NMS not available')}")
                    print(f"   torch: {probe_result.get('torch_version', 'unknown')}")
                    print(f"   torchvision: {probe_result.get('torchvision_version', 'unknown')}")
                    print(f"   NMS available: {probe_result.get('torchvision_nms_available', False)}")

                    # Attempt automatic rebuild
                    if torch_ver:
                        print("\n🔄 Attempting automatic torchvision rebuild with CUDA extensions...")
                        rebuild_success = _rebuild_torchvision_jetson(
                            venv_python,
                            venv_pip,
                            torch_ver,
                            _jetson_cuda_env()
                        )

                        if rebuild_success:
                            # Verify again after rebuild
                            print("\n   Re-verifying torchvision after rebuild...")
                            probe_result = _probe_torch_torchvision(venv_python, _jetson_cuda_env())

                            if probe_result.get("torchvision_nms_available"):
                                print(f"✅ Torchvision rebuild successful!")
                                print(f"   torch: {probe_result.get('torch_version')}")
                                print(f"   torchvision: {probe_result.get('torchvision_version')}")
                                print(f"   NMS available: {probe_result.get('torchvision_nms_available')}")
                            else:
                                print("⚠️  Torchvision rebuild completed but NMS still not available")
                                print(f"   Error: {probe_result.get('error', 'unknown')}")
                        else:
                            print("❌ Automatic torchvision rebuild failed")
                            print("   You may need to build it manually:")
                            print(f"     source venv/bin/activate")
                            print(f"     pip install --no-cache-dir --no-deps git+https://github.com/pytorch/vision.git@{vision_tag}")
                    else:
                        print("⚠️  Cannot determine torch version for torchvision rebuild")
                else:
                    print(f"✅ Torchvision verification passed!")
                    print(f"   torch: {probe_result.get('torch_version')}")
                    print(f"   torchvision: {probe_result.get('torchvision_version')}")
                    print(f"   NMS available: {probe_result.get('torchvision_nms_available')}")

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
                    error_output = probe_res.stderr.strip() or probe_res.stdout.strip()

                    # Check for missing library errors
                    if "libcusparseLt.so" in error_output or "cannot open shared object file" in error_output:
                        print("❌ PyTorch is missing required CUDA libraries.")
                        print("   The automatic cuSPARSELt installation may have failed.")
                        print("\n   SOLUTIONS:")
                        print("   1. Install cuSPARSELt manually (recommended):")
                        print("      wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb")
                        print("      sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb")
                        print("      sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/")
                        print("      sudo apt-get update")
                        print("      sudo apt-get -y install libcusparselt0 libcusparselt-dev")
                        print("      rm -rf venv/.deps_installed && python3 app.py")
                        print("\n   2. Or download from: https://developer.nvidia.com/cusparselt-downloads")
                        print(f"\n   Error details: {error_output[:200]}")
                    else:
                        print("❌ CUDA not available after installing Jetson torch.")
                        print("   This likely means the installed wheel is CPU-only or CUDA drivers are not visible.")
                        print("   If you overrode versions, set JETSON_TORCH_WHEEL to a CUDA-enabled wheel URL.")
                        print("   Otherwise, check JetPack/CUDA installation on the device.")
                        print(f"   Probe output: {error_output[:200]}")

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

# Check for torchvision NMS availability and provide helpful error message
try:
    import torchvision  # noqa: F401
    _TORCHVISION_NMS_AVAILABLE = hasattr(torch.ops.torchvision, "nms")
    if not _TORCHVISION_NMS_AVAILABLE:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: torchvision::nms operator not available")
        print("=" * 70)
        print("This usually means torchvision was not built with CUDA extensions.")
        print("\nTo fix this issue:")
        if _is_jetson():
            print("1. Delete the installation marker: rm venv/.deps_installed")
            print("2. Re-run the app: python3 app.py")
            print("   (The app will automatically rebuild torchvision with CUDA)")
        else:
            print("1. Reinstall torchvision with proper CUDA support")
            print("2. Or build from source: pip install --no-deps git+https://github.com/pytorch/vision.git")
        print("\nAttempting to continue anyway...")
        print("=" * 70 + "\n")
except Exception as e:
    print(f"⚠️  Warning: Could not check torchvision NMS availability: {e}")
    _TORCHVISION_NMS_AVAILABLE = False

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.registry import MODEL_REGISTRY
except ImportError as e:
    print(f"❌ Error importing DA3 modules: {e}")

    # Provide more specific guidance for the NMS error
    if "torchvision::nms" in str(e) or "operator torchvision::nms does not exist" in str(e):
        print("\n" + "=" * 70)
        print("🔧 DETECTED ISSUE: torchvision C++ extensions not compiled")
        print("=" * 70)
        if _is_jetson():
            print("\nAutomatic fix steps:")
            print("1. Delete: rm venv/.deps_installed")
            print("2. Re-run: python3 app.py")
            print("\nThe bootstrap will automatically rebuild torchvision with CUDA.")
        else:
            print("\nFix by rebuilding torchvision:")
            print("  source venv/bin/activate")
            print("  pip uninstall -y torchvision")
            print("  pip install --no-deps git+https://github.com/pytorch/vision.git")
        print("=" * 70)

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
        self.current_video_sequence = None  # Store video frame sequences
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
                frame_data_list = []  # Store per-frame data for video playback

                confidence_maps = None
                if apply_confidence_filter or include_confidence:
                    confidence_maps = getattr(prediction, "conf", None)

                # Extract camera poses if available (for video sequences)
                print(f"\n🔍 Camera Pose Prediction Debug:")
                print(f"  • Prediction type: {type(prediction).__name__}")
                print(f"  • Has 'extrinsics' attr: {hasattr(prediction, 'extrinsics')}")
                print(f"  • Has 'intrinsics' attr: {hasattr(prediction, 'intrinsics')}")
                print(f"  • Available attrs: {[a for a in dir(prediction) if not a.startswith('_')]}")

                camera_poses = getattr(prediction, "extrinsics", None)
                print(f"  • prediction.extrinsics value: {type(camera_poses).__name__ if camera_poses is not None else 'None'}")
                if camera_poses is not None:
                    print(f"  • Number of poses: {len(camera_poses) if hasattr(camera_poses, '__len__') else 'N/A'}")
                    if hasattr(camera_poses, '__len__') and len(camera_poses) > 0:
                        first_pose = np.array(camera_poses[0])
                        print(f"  • First pose shape: {first_pose.shape}")
                        print(f"  • First pose is identity: {np.allclose(first_pose, np.eye(4) if first_pose.shape == (4,4) else np.eye(3,4))}")
                        if first_pose.shape[0] >= 3 and first_pose.shape[1] >= 4:
                            print(f"  • First pose translation: {first_pose[:3, 3]}")
                else:
                    print(f"  ⚠️  Model did NOT predict camera poses - using identity matrices")
                    print(f"  ⚠️  This will result in all camera positions at (0,0,0)")
                print()

                if camera_poses is None:
                    # Use identity matrices as fallback (will show all at origin)
                    camera_poses = [np.eye(4) for _ in range(len(prediction.depth))]

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

                    # Transform points from camera frame to world frame using extrinsics (assumed world->camera)
                    cam_pose = camera_poses[i] if i < len(camera_poses) else np.eye(4, dtype=np.float32)
                    cam_to_world = None
                    try:
                        cam_pose = np.array(cam_pose, dtype=np.float32)
                        if cam_pose.shape == (4, 4):
                            cam_to_world = np.linalg.inv(cam_pose)  # invert to get camera->world
                            homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
                            points_world = (cam_to_world @ homog.T).T[:, :3]
                        else:
                            points_world = points
                    except Exception:
                        points_world = points

                    points_list.append(points_world)
                    colors_list.append(colors)
                    if conf_flat is not None:
                        conf_list.append(conf_flat)

                    # Store per-frame data for video sequences
                    if is_video:
                        # Downsample points for individual frames if needed
                        frame_points = points_world
                        frame_colors = colors
                        frame_conf = conf_flat

                        points_per_frame = max_points // len(prediction.depth) if max_points else len(frame_points)
                        if len(frame_points) > points_per_frame:
                            indices = np.random.choice(len(frame_points), points_per_frame, replace=False)
                            frame_points = frame_points[indices]
                            frame_colors = frame_colors[indices]
                            if frame_conf is not None:
                                frame_conf = frame_conf[indices]

                        frame_dict = {
                            "frame_index": i,
                            "vertices": frame_points.tolist(),
                            "colors": frame_colors.tolist(),
                            # Store camera-to-world so downstream viewers/path use correct orientation.
                            # Row-major (nested lists) is preserved for backward compatibility; flat column-major
                            # is added for Three.js clients that want direct Matrix4.fromArray usage.
                            "camera_pose": (cam_to_world.tolist() if cam_to_world is not None else (np.eye(4).tolist())),
                            "camera_pose_col_major_flat": (
                                cam_to_world.T.reshape(-1).tolist() if cam_to_world is not None else np.eye(4).T.reshape(-1).tolist()
                            ),
                            "intrinsics": ixt.tolist(),
                            "num_points": len(frame_points)
                        }
                        if frame_conf is not None:
                            frame_dict["confidence"] = frame_conf.tolist()
                        frame_data_list.append(frame_dict)

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
                        "filename": filename,
                        "is_video": is_video,
                        "num_frames": len(frame_data_list) if is_video else 1
                    }
                }
                if include_confidence and all_conf is not None:
                    state.current_pointcloud["confidence"] = all_conf.tolist()

                # Store video sequence data separately
                if is_video and frame_data_list:
                    state.current_video_sequence = {
                        "frames": frame_data_list,
                        "num_frames": len(frame_data_list),
                        "fps": feat_vis_fps if feat_vis_fps else 15,
                        "metadata": {
                            "filename": filename,
                            "resolution": resolution,
                            "total_points": len(all_points),
                            "points_per_frame": [f["num_points"] for f in frame_data_list]
                        }
                    }

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

@app.route('/api/video/info', methods=['GET'])
def get_video_info():
    """Get information about the current video sequence."""
    if state.current_video_sequence is None:
        return jsonify({"error": "No video sequence available"}), 404

    return jsonify({
        "num_frames": state.current_video_sequence["num_frames"],
        "fps": state.current_video_sequence["fps"],
        "metadata": state.current_video_sequence["metadata"]
    })

@app.route('/api/video/frame/<int:frame_index>', methods=['GET'])
def get_video_frame(frame_index):
    """Get a specific frame from the video sequence."""
    if state.current_video_sequence is None:
        return jsonify({"error": "No video sequence available"}), 404

    if frame_index < 0 or frame_index >= state.current_video_sequence["num_frames"]:
        return jsonify({"error": f"Frame index {frame_index} out of range [0, {state.current_video_sequence['num_frames']-1}]"}), 400

    frame_data = state.current_video_sequence["frames"][frame_index]
    return jsonify(frame_data)

@app.route('/api/video/frames', methods=['GET'])
def get_all_video_frames():
    """Get all frames from the video sequence."""
    if state.current_video_sequence is None:
        return jsonify({"error": "No video sequence available"}), 404

    # Optional: Support range queries via query params
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', state.current_video_sequence["num_frames"], type=int)

    frames = state.current_video_sequence["frames"][start:end]
    return jsonify({
        "frames": frames,
        "start": start,
        "end": min(end, state.current_video_sequence["num_frames"]),
        "total": state.current_video_sequence["num_frames"]
    })

@app.route('/api/video/camera_path', methods=['GET'])
def get_camera_path():
    """Get the camera path for visualization."""
    if state.current_video_sequence is None:
        return jsonify({"error": "No video sequence available"}), 404

    camera_poses = [frame["camera_pose"] for frame in state.current_video_sequence["frames"]]
    camera_poses_col_major_flat = []
    camera_poses_c2w = []  # Camera-to-world poses

    # DEBUG: Log first pose to check if it's valid
    if len(camera_poses) > 0:
        first_pose = np.array(camera_poses[0], dtype=np.float32)
        print("\n" + "="*60)
        print("🔍 Camera Path Debug - First Frame")
        print("="*60)
        print(f"First extrinsics (w2c) shape: {first_pose.shape}")
        print(f"First extrinsics:\n{first_pose}")
        print(f"Is identity? {np.allclose(first_pose, np.eye(4) if first_pose.shape == (4, 4) else np.eye(3, 4))}")
        if first_pose.shape == (4, 4):
            print(f"Translation component (w2c): {first_pose[:3, 3]}")
            try:
                c2w_test = np.linalg.inv(first_pose)
                print(f"Inverted c2w translation: {c2w_test[:3, 3]}")
            except:
                print("ERROR: Cannot invert matrix!")
        print("="*60 + "\n")

    # REMOVED: Synthetic path generation - use REAL poses from DA3 model!

    # Extract camera positions from poses for path visualization
    camera_positions = []
    for pose in camera_poses:
        # pose is 4x4 extrinsics matrix (world-to-camera)
        # We need to invert it to get the camera position in world space
        pose_array = np.array(pose, dtype=np.float32)

        # Invert extrinsics to get camera-to-world
        try:
            c2w = np.linalg.inv(pose_array)
            # Camera position in world space is the translation of c2w
            position = c2w[:3, 3].tolist()
            camera_positions.append(position)

            # Store c2w for frustum visualization
            camera_poses_c2w.append(c2w.tolist())

            # Provide column-major flattening of c2w (Three.js Matrix4.fromArray-friendly)
            camera_poses_col_major_flat.append(c2w.T.reshape(-1).tolist())
        except Exception as e:
            print(f"Warning: Failed to invert camera pose: {e}")
            # Fallback: use identity
            position = [0.0, 0.0, 0.0]
            camera_positions.append(position)
            camera_poses_col_major_flat.append(None)
            camera_poses_c2w.append(pose_array.tolist())

    return jsonify({
        "poses": camera_poses,  # Original extrinsics (world-to-camera)
        "c2w_poses": camera_poses_c2w,  # Inverted to camera-to-world
        "poses_col_major_flat": camera_poses_col_major_flat,  # c2w in column-major format
        "c2w_poses_col_major_flat": camera_poses_col_major_flat,  # Same as above, explicit name
        "positions": camera_positions,  # Extracted from c2w matrices
        "num_frames": len(camera_poses),
        "pose_convention": {
            "type": "camera_to_world",
            "storage": "c2w_poses are 4x4 row-major; camera position in last column",
            "column_major_flat": "poses_col_major_flat provides c2w transform flattened column-major for Three.js Matrix4.fromArray"
        }
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
    print("  • /api/video/info")
    print("  • /api/video/frame/<frame_index>")
    print("  • /api/video/frames")
    print("  • /api/video/camera_path")
    print("  • /api/export/glb")
    print("  • /api/v1/* aliases")
    print("\n⚠️  Press Ctrl+C to stop the server")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
