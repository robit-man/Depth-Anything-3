#!/usr/bin/env python3
"""
Example API Client for Depth Anything 3 RESTful API
Demonstrates how to use the API endpoints programmatically
"""

import requests
import base64
import json
import time
from pathlib import Path

# API Base URL
API_BASE = "http://localhost:5000"

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 data URI."""
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    # Detect image format
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')

    return f"data:{mime_type};base64,{img_data}"

def encode_video_to_base64(video_path: str) -> str:
    """Encode video file to base64 data URI."""
    with open(video_path, 'rb') as f:
        video_data = base64.b64encode(f.read()).decode()
    return f"data:video/mp4;base64,{video_data}"

# ============================================================================
# Example 1: Process image from base64
# ============================================================================

def example_1_process_image_base64(image_path: str):
    """Process an image using base64 encoding."""
    print("=" * 70)
    print("Example 1: Process image from base64")
    print("=" * 70)

    # Encode image
    print(f"Encoding image: {image_path}")
    image_base64 = encode_image_to_base64(image_path)

    # Send request
    print("Sending request to API...")
    start_time = time.time()

    response = requests.post(f"{API_BASE}/api/v1/process/image", json={
        "image": image_base64,
        "resolution": 504,
        "max_points": 500000,
        "export_format": "json",
        "include_cameras": True,
        "include_confidence": False
    })

    elapsed = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! (took {elapsed:.2f}s)")
        print(f"  Number of points: {result['point_cloud']['metadata']['num_points']}")
        print(f"  Number of cameras: {len(result.get('cameras', []))}")
        print(f"  Bounds: {result['point_cloud']['metadata']['bounds']}")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.json())
        return None

# ============================================================================
# Example 2: Process image from URL
# ============================================================================

def example_2_process_image_url(url: str):
    """Process an image from a URL."""
    print("\n" + "=" * 70)
    print("Example 2: Process image from URL")
    print("=" * 70)

    print(f"Processing URL: {url}")

    response = requests.post(f"{API_BASE}/api/v1/process/image", json={
        "image_url": url,
        "resolution": 504,
        "export_format": "json"
    })

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  Number of points: {result['point_cloud']['metadata']['num_points']}")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.json())
        return None

# ============================================================================
# Example 3: Download GLB file
# ============================================================================

def example_3_download_glb(image_path: str, output_path: str = "output.glb"):
    """Process image and download GLB file."""
    print("\n" + "=" * 70)
    print("Example 3: Download GLB point cloud")
    print("=" * 70)

    image_base64 = encode_image_to_base64(image_path)

    print(f"Requesting GLB format...")
    response = requests.post(f"{API_BASE}/api/v1/process/image", json={
        "image": image_base64,
        "resolution": 756,
        "max_points": 1000000,
        "export_format": "glb",
        "include_cameras": True
    })

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ GLB saved to: {output_path}")
        print(f"  File size: {len(response.content) / 1024 / 1024:.2f} MB")
        return output_path
    else:
        print(f"✗ Error: {response.status_code}")
        return None

# ============================================================================
# Example 4: Process video
# ============================================================================

def example_4_process_video(video_path: str):
    """Process a video and get point cloud."""
    print("\n" + "=" * 70)
    print("Example 4: Process video")
    print("=" * 70)

    video_base64 = encode_video_to_base64(video_path)

    print(f"Processing video: {video_path}")
    print("This may take a while...")

    response = requests.post(f"{API_BASE}/api/v1/process/video", json={
        "video": video_base64,
        "fps": 2.0,
        "resolution": 504,
        "max_points": 1000000,
        "export_format": "json"
    }, timeout=300)  # 5 minute timeout

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  Processed frames: {result['metadata']['num_frames']}")
        print(f"  Processing time: {result['metadata']['processing_time']:.2f}s")
        print(f"  Number of points: {result['point_cloud']['metadata']['num_points']}")
        print(f"  Number of cameras: {len(result['cameras'])}")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        return None

# ============================================================================
# Example 5: Process batch of images
# ============================================================================

def example_5_process_batch(image_paths: list):
    """Process multiple images in batch."""
    print("\n" + "=" * 70)
    print("Example 5: Process batch of images")
    print("=" * 70)

    images = []
    for img_path in image_paths:
        print(f"Encoding: {img_path}")
        images.append({
            "data": encode_image_to_base64(img_path)
        })

    print(f"Processing {len(images)} images...")

    response = requests.post(f"{API_BASE}/api/v1/process/batch", json={
        "images": images,
        "resolution": 504,
        "max_points": 1000000
    })

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  Number of points: {result['point_cloud']['metadata']['num_points']}")
        print(f"  Number of cameras: {len(result['cameras'])}")
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        return None

# ============================================================================
# Example 6: Convert to Three.js format
# ============================================================================

def example_6_threejs_format(image_path: str):
    """Get point cloud in Three.js-ready format."""
    print("\n" + "=" * 70)
    print("Example 6: Get Three.js format")
    print("=" * 70)

    # Get JSON result
    result = example_1_process_image_base64(image_path)

    if result:
        # Convert to Three.js BufferGeometry format
        threejs_data = {
            "positions": [],  # Flat array: [x, y, z, x, y, z, ...]
            "colors": []      # Flat array: [r/255, g/255, b/255, ...]
        }

        for vertex in result['point_cloud']['vertices']:
            threejs_data['positions'].extend(vertex)

        for color in result['point_cloud']['colors']:
            # Normalize to 0-1 for Three.js
            threejs_data['colors'].extend([c/255.0 for c in color])

        print(f"\n✓ Three.js format ready!")
        print(f"  Positions array length: {len(threejs_data['positions'])}")
        print(f"  Colors array length: {len(threejs_data['colors'])}")

        # Save to file
        with open('threejs_pointcloud.json', 'w') as f:
            json.dump(threejs_data, f)
        print(f"  Saved to: threejs_pointcloud.json")

        return threejs_data

# ============================================================================
# Example 7: Check API health
# ============================================================================

def example_7_health_check():
    """Check API health and status."""
    print("\n" + "=" * 70)
    print("Example 7: Health check")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/api/v1/health")

    if response.status_code == 200:
        status = response.json()
        print(f"✓ API is healthy")
        print(f"  Model loaded: {status['model_loaded']}")
        print(f"  Model name: {status['model_name']}")
        print(f"  CUDA available: {status['cuda_available']}")
        return status
    else:
        print(f"✗ API unhealthy: {response.status_code}")
        return None

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Depth Anything 3 API")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--url", type=str, help="URL to test image")
    parser.add_argument("--example", type=int, choices=range(1, 8), help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples (requires --image)")

    args = parser.parse_args()

    # Check health first
    example_7_health_check()

    if args.all and args.image:
        # Run all examples
        example_1_process_image_base64(args.image)
        example_3_download_glb(args.image)
        example_6_threejs_format(args.image)

    elif args.example == 1 and args.image:
        example_1_process_image_base64(args.image)

    elif args.example == 2 and args.url:
        example_2_process_image_url(args.url)

    elif args.example == 3 and args.image:
        example_3_download_glb(args.image)

    elif args.example == 4 and args.video:
        example_4_process_video(args.video)

    elif args.example == 5 and args.image:
        # Use same image multiple times for demo
        example_5_process_batch([args.image] * 3)

    elif args.example == 6 and args.image:
        example_6_threejs_format(args.image)

    elif args.example == 7:
        # Already ran above
        pass

    else:
        print("\n📖 Usage examples:")
        print("  python3 example_api_client.py --example 1 --image test.jpg")
        print("  python3 example_api_client.py --example 2 --url https://example.com/image.jpg")
        print("  python3 example_api_client.py --example 3 --image test.jpg")
        print("  python3 example_api_client.py --example 4 --video test.mp4")
        print("  python3 example_api_client.py --all --image test.jpg")
        print("\n💡 Make sure the Flask server is running: python3 main.py")
