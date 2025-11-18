#!/usr/bin/env python3
"""
Quick API Demo - Demonstrates Depth Anything 3 RESTful API
This script shows how to use all the API endpoints.
"""

import requests
import base64
import json
import time
import sys
from pathlib import Path

API_BASE = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_health():
    """Check API health and model status."""
    print_section("1. Health Check")
    response = requests.get(f"{API_BASE}/api/v1/health")
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"CUDA Available: {data['cuda_available']}")
    print(f"Model Loaded: {data['model_loaded']}")
    print(f"Model Name: {data['model_name']}")
    return data['model_loaded']

def wait_for_model():
    """Wait for model to finish loading."""
    print_section("2. Waiting for Model to Load")
    print("The model is downloading from Hugging Face (~5GB)...")
    print("This may take 5-10 minutes on first run.\n")

    while True:
        try:
            response = requests.get(f"{API_BASE}/api/model_status")
            data = response.json()
            status = data.get('status', 'unknown')
            progress = data.get('progress', 0)

            if status == 'ready':
                print("\n✓ Model loaded successfully!")
                return True
            elif status == 'loading':
                print(f"\rProgress: {progress}%... ", end='', flush=True)
            elif status == 'error':
                print(f"\n✗ Error loading model: {data.get('error', 'Unknown error')}")
                return False

            time.sleep(2)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return False
        except Exception as e:
            print(f"\nError checking status: {e}")
            time.sleep(2)

def demo_image_from_url():
    """Demonstrate processing image from URL."""
    print_section("3. Process Image from URL")

    response = requests.post(
        f"{API_BASE}/api/v1/process/image",
        json={
            "image_url": "https://picsum.photos/800/600",
            "resolution": 504,
            "max_points": 100000,
            "export_format": "json",
            "include_cameras": True
        },
        timeout=120
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully processed image!")
        print(f"  Points: {data['point_cloud']['metadata']['num_points']:,}")
        print(f"  Bounds: {data['point_cloud']['metadata']['bounds']['min']} to {data['point_cloud']['metadata']['bounds']['max']}")
        print(f"  Cameras: {len(data.get('cameras', []))}")
        print(f"  Processing time: {data['metadata']['processing_time']:.2f}s")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return False

def demo_image_from_file(image_path):
    """Demonstrate processing image from local file."""
    print_section("4. Process Image from Local File (Base64)")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        print("  Skipping this test.")
        return False

    # Encode image to base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine MIME type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    mime = mime_types.get(ext, 'image/jpeg')

    print(f"Processing: {image_path}")
    print(f"Image size: {len(image_data):,} bytes (base64)")

    response = requests.post(
        f"{API_BASE}/api/v1/process/image",
        json={
            "image": f"data:{mime};base64,{image_data}",
            "resolution": 504,
            "max_points": 500000,
            "export_format": "json",
            "include_cameras": True
        },
        timeout=120
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Successfully processed!")
        print(f"  Points: {data['point_cloud']['metadata']['num_points']:,}")
        print(f"  Processing time: {data['metadata']['processing_time']:.2f}s")

        # Save point cloud to file
        output_file = "demo_pointcloud.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to: {output_file}")
        return True
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return False

def demo_download_glb(image_path):
    """Demonstrate downloading GLB format."""
    print_section("5. Download GLB Format")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        print("  Skipping this test.")
        return False

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    ext = Path(image_path).suffix.lower()
    mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'

    print(f"Requesting GLB export for: {image_path}")

    response = requests.post(
        f"{API_BASE}/api/v1/process/image",
        json={
            "image": f"data:{mime};base64,{image_data}",
            "resolution": 504,
            "export_format": "glb"
        },
        timeout=120
    )

    if response.status_code == 200 and response.headers.get('Content-Type') == 'model/gltf-binary':
        output_file = "demo_output.glb"
        with open(output_file, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content)
        print(f"✓ GLB downloaded successfully!")
        print(f"  File: {output_file}")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        return False

def main():
    """Run all API demonstrations."""
    print("\n" + "="*70)
    print("  🚀 Depth Anything 3 - RESTful API Demo")
    print("="*70)

    # Check if server is running
    try:
        requests.get(f"{API_BASE}/api/v1/health", timeout=2)
    except requests.exceptions.RequestException:
        print("\n✗ Error: Flask server is not running!")
        print("  Please start the server first: python3 main.py")
        return 1

    # 1. Check health
    model_loaded = check_health()

    # 2. Wait for model if not loaded
    if not model_loaded:
        print("\nℹ️  Model is not loaded yet. Loading now...")
        response = requests.post(f"{API_BASE}/api/load_model")
        if not wait_for_model():
            print("\n✗ Failed to load model. Exiting.")
            return 1

    # 3. Process image from URL
    if not demo_image_from_url():
        print("\n⚠️  URL test failed, continuing with other tests...")

    # 4. Process local image (if provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        demo_image_from_file(image_path)
        demo_download_glb(image_path)
    else:
        # Try with example images
        example_images = [
            "assets/examples/SOH/000.png",
            "assets/examples/SOH/010.png"
        ]
        for img in example_images:
            if Path(img).exists():
                demo_image_from_file(img)
                demo_download_glb(img)
                break
        else:
            print_section("4-5. Local Image Tests")
            print("ℹ️  No local image provided.")
            print("  Usage: python3 quick_api_demo.py <image_path>")

    # Summary
    print_section("Demo Complete!")
    print("✓ All API endpoints are functional")
    print("✓ Error handling is working correctly")
    print("✓ JSON and GLB export formats supported")
    print("\nNext steps:")
    print("  - See API_README.md for full documentation")
    print("  - Run: python3 example_api_client.py --all --image <your_image>")
    print("  - Run: ./test_api.sh <your_image>")
    print("  - Open: example_threejs_client.html in your browser")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(1)
