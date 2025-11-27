# Critical Fix Applied: Torchvision Build Isolation Issue

## Problem Identified

The torchvision build was failing with:
```
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

This occurred because pip's build isolation creates a clean environment that doesn't inherit our `CUDA_HOME` environment variable, even though we set it in the `env` parameter.

## Root Cause

When using pip to install from source (e.g., `git+https://github.com/pytorch/vision.git@v0.19.0`), pip creates a build isolation environment that:

1. Only includes packages listed in `pyproject.toml` build dependencies
2. **Does NOT inherit environment variables** from the parent process
3. This meant our carefully set `CUDA_HOME`, `FORCE_CUDA`, and `TORCH_CUDA_ARCH_LIST` were invisible to the build

## Fixes Applied

### 1. Use `--no-build-isolation` Flag

**Location**: Lines 462, 916 in [app.py](app.py)

Changed from:
```python
subprocess.run([
    str(venv_pip), "install", "--no-cache-dir", "--force-reinstall", "--no-deps",
    f"git+https://github.com/pytorch/vision.git@{vision_tag}",
], env=build_env, ...)
```

To:
```python
subprocess.run([
    str(venv_pip), "install", "--no-cache-dir", "--force-reinstall", "--no-deps",
    "--no-build-isolation",  # Critical: allows CUDA_HOME to be seen
    f"git+https://github.com/pytorch/vision.git@{vision_tag}",
], env=build_env, ...)
```

### 2. Install numpy Before Build

**Location**: Lines 428, 885 in [app.py](app.py)

The build was also failing because numpy wasn't available in the build environment. Now we install it first:

```python
subprocess.run(
    [str(venv_pip), "install", "--upgrade", "setuptools", "wheel", "ninja", "numpy"],
    check=False,
    capture_output=True,
    env=cuda_env
)
```

### 3. Don't Uninstall torchvision Before Build

**Location**: Line 437 in [app.py](app.py)

Changed behavior to NOT uninstall the existing (broken) torchvision until after the rebuild succeeds. This prevents leaving the system with no torchvision at all if the build fails.

Before:
```python
# Uninstall existing torchvision
subprocess.run([str(venv_pip), "uninstall", "-y", "torchvision"], ...)
# Build torchvision (if this fails, now there's NO torchvision)
```

After:
```python
# DON'T uninstall torchvision yet - only if build succeeds
# This prevents leaving the system with no torchvision at all

# Build torchvision with --force-reinstall (replaces old one)
```

### 4. Fixed JetPack Version Detection

**Location**: Lines 260-266 in [app.py](app.py)

Fixed the mapping from R-version to JetPack version:

Before:
```python
info["jetpack_version"] = f"JP{rev_match.group(1)[0]}.x"  # R36 -> JP3.x (wrong!)
```

After:
```python
r_version = int(rev_match.group(1))
jp_version = r_version - 30  # R36->6, R35->5, R34->4
info["jetpack_version"] = f"JP{jp_version}.x"  # R36 -> JP6.x (correct!)
```

## How to Test

On your Jetson, run:

```bash
cd ~/Depth-Anything-3
rm venv/.deps_installed
python3 app.py
```

Expected output:
1. ✅ System diagnostics show "JetPack: JP6.x" (not JP3.x)
2. ✅ Detects torchvision without NMS
3. ✅ Automatically rebuilds with `--no-build-isolation`
4. ✅ Build succeeds (CUDA_HOME is visible)
5. ✅ NMS verification passes
6. ✅ App starts successfully

## What `--no-build-isolation` Does

From pip documentation:

> `--no-build-isolation`: Disable isolation when building a modern source distribution. Build dependencies specified by PEP 518 must be already installed if this option is used.

This means:
- The build uses the venv's installed packages (setuptools, wheel, ninja, numpy)
- **Environment variables from the parent process are visible**
- Our `CUDA_HOME=/usr/local/cuda-12.2` is now visible to the torchvision build
- Our `FORCE_CUDA=1` tells the build to compile CUDA extensions
- Our `TORCH_CUDA_ARCH_LIST=8.7` targets the AGX Orin GPU

## Technical Details

The torchvision `setup.py` uses `torch.utils.cpp_extension.CUDAExtension` which calls `_join_cuda_home()`:

```python
def _join_cuda_home(*paths):
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME environment variable is not set. '
                     'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)
```

Without `--no-build-isolation`, `CUDA_HOME` from our environment never reaches this code. With the flag, it does!

## Files Modified

- [app.py](app.py): All fixes applied
  - Line 263-266: Fixed JetPack version detection
  - Line 428: Install numpy before build (rebuild function)
  - Line 437: Don't pre-uninstall torchvision
  - Line 462: Add `--no-build-isolation` (rebuild function)
  - Line 885: Install numpy before build (initial install)
  - Line 916: Add `--no-build-isolation` (initial install)

## Expected Build Output

With these fixes, you should see:

```
🔨 Rebuilding torchvision from source with CUDA extensions...
   Torch version: 2.4.0
   Target torchvision tag: v0.19.0
   Using CUDA_HOME: /usr/local/cuda-12.2
   Using TORCH_CUDA_ARCH_LIST: 8.7
   Installing system build dependencies...
   Installing Python build tools...
   Building torchvision from v0.19.0...

Building wheel torchvision-0.19.0a0+48b1edf
Compiling extensions with following flags:
  FORCE_CUDA: True      <-- ✅ CUDA enabled!
  NVCC_FLAGS: -gencode arch=compute_87,code=sm_87  <-- ✅ AGX Orin arch!

[15-30 minutes of compilation...]

✓ Torchvision built and installed successfully

✅ Torchvision verification passed!
   torch: 2.4.0a0+3bcc3cddb5.nv24.07
   torchvision: 0.19.0a0+48b1edf
   NMS available: True   <-- ✅ Success!
```

The key indicators of success:
- `FORCE_CUDA: True` (not False)
- `NVCC_FLAGS` includes the correct architecture
- NMS verification passes

Run it now and it should work! 🚀
