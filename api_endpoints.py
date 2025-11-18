"""
RESTful API endpoints for Depth Anything 3
Provides programmatic access to point cloud generation
"""

import base64
import io
import time
import numpy as np
from PIL import Image
from flask import jsonify, request, send_file
from pathlib import Path

# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        # Handle data URI format
        if data.startswith('data:'):
            data = data.split(',')[1]

        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def decode_base64_video(data: str, output_path: str):
    """Decode base64 video and save to file."""
    try:
        if data.startswith('data:'):
            data = data.split(',')[1]

        video_bytes = base64.b64decode(data)
        with open(output_path, 'wb') as f:
            f.write(video_bytes)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 video: {e}")

def download_from_url(url: str, output_path: str = None) -> str:
    """Download file from URL."""
    try:
        import requests
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        if output_path is None:
            # Return as numpy array for images
            img = Image.open(io.BytesIO(response.content))
            return np.array(img)
        else:
            # Save to file for videos
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
    except Exception as e:
        raise ValueError(f"Failed to download from URL: {e}")

def prediction_to_point_cloud_json(prediction, max_points: int = None, include_confidence: bool = False) -> dict:
    """Convert DA3 prediction to JSON point cloud format."""
    try:
        # Extract point cloud data from depth maps
        points_list = []
        colors_list = []
        conf_list = []

        for i in range(len(prediction.depth)):
            depth = prediction.depth[i]
            image = prediction.processed_images[i]
            intrinsics = prediction.intrinsics[i]
            extrinsics = prediction.extrinsics[i]

            # Create point cloud from depth
            h, w = depth.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]

            # Unproject to 3D
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            x3d = -(x_coords - cx) * depth / fx  # Negate X to match camera convention
            y3d = -(y_coords - cy) * depth / fy  # Negate Y to convert from image coords (Y down) to 3D coords (Y up)
            z3d = depth

            # Stack to get points in camera coordinates
            points_cam = np.stack([x3d, y3d, z3d], axis=-1).reshape(-1, 3)
            colors = image.reshape(-1, 3)

            # Transform to world coordinates
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3] if extrinsics.shape[1] == 4 else extrinsics[:3, 3]

            # Convert from camera to world: P_world = R^T * (P_cam - t)
            points_world = (R.T @ (points_cam.T - t.reshape(3, 1))).T

            points_list.append(points_world)
            colors_list.append(colors)

            if include_confidence and hasattr(prediction, 'conf'):
                conf_list.append(prediction.conf[i].reshape(-1))

        # Combine all points
        all_points = np.vstack(points_list)
        all_colors = np.vstack(colors_list)

        # Filter invalid points (inf, nan, zero/negative depth)
        valid_mask = np.isfinite(all_points).all(axis=1) & (all_points[:, 2] > 0.01)
        all_points = all_points[valid_mask]
        all_colors = all_colors[valid_mask]

        # Downsample if needed
        if max_points and len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]

        result = {
            "vertices": all_points.tolist(),
            "colors": all_colors.tolist(),
            "metadata": {
                "num_points": int(len(all_points)),
                "coordinate_system": "opengl",
                "units": "arbitrary",
                "bounds": {
                    "min": all_points.min(axis=0).tolist(),
                    "max": all_points.max(axis=0).tolist(),
                    "center": all_points.mean(axis=0).tolist()
                }
            }
        }

        if include_confidence and conf_list:
            all_conf = np.hstack(conf_list)[valid_mask]
            if max_points and len(all_conf) > max_points:
                all_conf = all_conf[indices]
            result["confidence"] = all_conf.tolist()

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to convert prediction to point cloud: {e}")

def extract_camera_data(prediction) -> list:
    """Extract camera parameters from prediction."""
    cameras = []

    for i in range(len(prediction.extrinsics)):
        ext = prediction.extrinsics[i]  # (3x4) or (4x4)
        ixt = prediction.intrinsics[i]  # (3x3)

        # Extract rotation and translation
        R = ext[:3, :3]
        t = ext[:3, 3] if ext.shape[1] == 4 else ext[:3, 3]

        # Camera position in world coords: -R^T * t
        camera_pos = -R.T @ t

        cameras.append({
            "id": int(i),
            "type": "perspective",
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
                "translation": t.tolist()
            }
        })

    return cameras

# ============================================================================
# API ROUTE CREATORS
# ============================================================================

def create_api_routes(app, state):
    """Create API routes for the Flask app."""

    @app.route('/api/v1/process/image', methods=['POST'])
    def api_process_image():
        """Process single image and return point cloud data."""
        if not state.model_ready:
            return jsonify({'error': 'Model not loaded. Call /api/load_model first.'}), 503

        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            # Validate input
            if not any(k in data for k in ['image', 'image_url', 'image_path']):
                return jsonify({'error': 'No image provided. Use "image" (base64), "image_url", or "image_path"'}), 400

            start_time = time.time()

            # Load image
            if 'image' in data:
                img = decode_base64_image(data['image'])
            elif 'image_url' in data:
                img = download_from_url(data['image_url'])
            else:
                img_path = data['image_path']
                if not Path(img_path).exists():
                    return jsonify({'error': f'Image path does not exist: {img_path}'}), 404
                img = np.array(Image.open(img_path))

            # Get parameters
            resolution = int(data.get('resolution', 504))
            max_points = int(data.get('max_points', 1000000))
            export_format = data.get('export_format', 'json')
            include_cameras = data.get('include_cameras', True)
            include_confidence = data.get('include_confidence', False)

            # Process with model
            prediction = state.model.inference(
                image=[img],
                process_res=resolution,
                num_max_points=max_points
            )

            processing_time = time.time() - start_time

            # Format response based on export_format
            if export_format == 'json':
                point_cloud = prediction_to_point_cloud_json(
                    prediction,
                    max_points=max_points,
                    include_confidence=include_confidence
                )

                result = {
                    "success": True,
                    "point_cloud": point_cloud,
                    "metadata": {
                        "processing_time": round(processing_time, 2),
                        "resolution": resolution,
                        "max_points": max_points
                    }
                }

                if include_cameras:
                    result["cameras"] = extract_camera_data(prediction)

                return jsonify(result)

            elif export_format == 'glb':
                # Export to GLB and return binary
                output_dir = Path(app.config['OUTPUT_FOLDER']) / f"api_{int(time.time())}"
                output_dir.mkdir(exist_ok=True)

                from depth_anything_3.utils.export.glb import export_to_glb
                glb_path = export_to_glb(
                    prediction,
                    str(output_dir),
                    num_max_points=max_points,
                    show_cameras=include_cameras
                )

                return send_file(
                    glb_path,
                    mimetype='model/gltf-binary',
                    as_attachment=True,
                    download_name='point_cloud.glb'
                )

            else:
                return jsonify({'error': f'Unsupported export format: {export_format}'}), 400

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    @app.route('/api/v1/process/video', methods=['POST'])
    def api_process_video():
        """Process video and return point cloud data."""
        if not state.model_ready:
            return jsonify({'error': 'Model not loaded'}), 503

        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            # Validate input
            if not any(k in data for k in ['video', 'video_url', 'video_path']):
                return jsonify({'error': 'No video provided'}), 400

            start_time = time.time()

            # Save video to temp file
            temp_video_path = Path(app.config['UPLOAD_FOLDER']) / f"api_video_{int(time.time())}.mp4"

            if 'video' in data:
                decode_base64_video(data['video'], str(temp_video_path))
            elif 'video_url' in data:
                download_from_url(data['video_url'], str(temp_video_path))
            else:
                temp_video_path = Path(data['video_path'])
                if not temp_video_path.exists():
                    return jsonify({'error': 'Video path does not exist'}), 404

            # Extract frames
            fps = float(data.get('fps', 2.0))
            resolution = int(data.get('resolution', 504))
            max_points = int(data.get('max_points', 1000000))
            export_format = data.get('export_format', 'json')

            # Use moviepy to extract frames
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(temp_video_path))
            duration = clip.duration
            frame_times = np.arange(0, duration, 1.0 / fps)

            frames = []
            for t in frame_times:
                frame = clip.get_frame(t)
                frames.append(frame)

            clip.close()

            # Process frames
            prediction = state.model.inference(
                image=frames,
                process_res=resolution,
                num_max_points=max_points
            )

            processing_time = time.time() - start_time

            # Format response
            if export_format == 'json':
                point_cloud = prediction_to_point_cloud_json(prediction, max_points=max_points)

                result = {
                    "success": True,
                    "point_cloud": point_cloud,
                    "cameras": extract_camera_data(prediction),
                    "metadata": {
                        "processing_time": round(processing_time, 2),
                        "num_frames": len(frames),
                        "fps": fps,
                        "duration": duration
                    }
                }

                return jsonify(result)

            elif export_format == 'glb':
                output_dir = Path(app.config['OUTPUT_FOLDER']) / f"api_video_{int(time.time())}"
                output_dir.mkdir(exist_ok=True)

                from depth_anything_3.utils.export.glb import export_to_glb
                glb_path = export_to_glb(prediction, str(output_dir), num_max_points=max_points)

                return send_file(
                    glb_path,
                    mimetype='model/gltf-binary',
                    as_attachment=True,
                    download_name='point_cloud.glb'
                )

        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    @app.route('/api/v1/process/batch', methods=['POST'])
    def api_process_batch():
        """Process multiple images in batch."""
        if not state.model_ready:
            return jsonify({'error': 'Model not loaded'}), 503

        try:
            data = request.json
            if not data or 'images' not in data:
                return jsonify({'error': 'No images array provided'}), 400

            images_data = data['images']
            if not isinstance(images_data, list):
                return jsonify({'error': 'images must be an array'}), 400

            start_time = time.time()

            # Load all images
            images = []
            for img_data in images_data:
                if 'data' in img_data:
                    img = decode_base64_image(img_data['data'])
                elif 'url' in img_data:
                    img = download_from_url(img_data['url'])
                elif 'path' in img_data:
                    img = np.array(Image.open(img_data['path']))
                else:
                    continue
                images.append(img)

            if not images:
                return jsonify({'error': 'No valid images provided'}), 400

            # Process
            resolution = int(data.get('resolution', 504))
            max_points = int(data.get('max_points', 1000000))

            prediction = state.model.inference(
                image=images,
                process_res=resolution,
                num_max_points=max_points
            )

            processing_time = time.time() - start_time

            # Return JSON format
            point_cloud = prediction_to_point_cloud_json(prediction, max_points=max_points)

            result = {
                "success": True,
                "point_cloud": point_cloud,
                "cameras": extract_camera_data(prediction),
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "num_images": len(images)
                }
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    @app.route('/api/v1/health', methods=['GET'])
    def api_health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "model_loaded": state.model_ready,
            "model_name": state.model_name,
            "cuda_available": __import__('torch').cuda.is_available()
        })

    print("✓ API routes registered")
