# Depth Anything 3 HTTP API

This document describes the REST endpoints exposed by `main.py` for running inference and managing models. The API follows simple JSON/multipart patterns and includes alias routes in an “Ollama-style” `/api/v1` namespace.

Base URL defaults to `http://localhost:<port>` where `<port>` is printed when the Flask app starts.

## Health
- `GET /api/v1/health`
  - Returns service heartbeat and model state.
  - Response
    ```json
    {
      "status": "ok",
      "model": { "id": "da3nested-giant-large", "ready": true, "loading": false, "downloaded": true },
      "active_jobs": 1
    }
    ```

## Models (list, download, load)
- `GET /api/models/list` (alias: `GET /api/v1/models`)
  - Lists available models with download/loading information.
  - Response: `{ "models": [ { "id", "name", "description", "size", "speed", "quality", "recommended_for", "downloaded", "current" }, ... ] }`

- `POST /api/models/select`
  - Body: `{ "model_id": "<id>" }`
  - Marks a model as selected (but does not load it). Response: `{ "message": "...", "status": optional }`

- `POST /api/load_model`
  - Loads the currently selected model asynchronously. Response: `{ "message": "Model loading started", "status": "loading" }`

- `POST /api/v1/models/download`
  - Body: `{ "model_id": "<id>" }`
  - Convenience alias: selects the model then invokes load; equivalent to `/api/models/select` + `/api/load_model`.

- `POST /api/v1/models/load`
  - Body: `{ "model_id": "<id>" }`
  - Explicitly select and load a model in one call.

- `GET /api/model_status`
  - Returns loading state for the currently selected model: `{ "status": "ready|loading|not_loaded", "progress": <0-100>, "model": "<id>" }`

## Inference / jobs
- `POST /api/process` (alias: `POST /api/v1/infer`)
  - Multipart form-data fields:
    - `file` (required): image or video file.
    - `resolution` (optional, int; default 504)
    - `max_points` (optional, int; default 1_000_000)
  - Starts async processing. Response: `{ "job_id": "job_<timestamp>", "message": "Processing started" }`

- `GET /api/job/<job_id>` (alias: `GET /api/v1/jobs/<job_id>`)`
  - Poll job status.
  - Response when complete: `{ "status": "completed", "pointcloud": { "vertices": [...], "colors": [...], "metadata": { ... } } }`
  - Response when error: `{ "status": "error", "error": "..." }`

## Floor alignment
- `POST /api/floor_align`
  - Aligns the current point cloud’s floor plane to `y=0`.
  - Response: `{ "message": "Floor alignment applied", "pointcloud": { ...updated... } }`

## Export
- `GET /api/export/glb` (alias: `GET /api/v1/export/glb`)`
  - Exports the current processed point cloud. Currently returns JSON payload of the point cloud (`export.json`) with vertices/colors/metadata.

## Notes for clients
Use `/api/v1/health` to gate availability before kicking off requests. A typical end-to-end flow against a local server at `http://127.0.0.1:5000`:

```bash
# 1) Health
curl http://127.0.0.1:5000/api/v1/health

# 2) List models
curl http://127.0.0.1:5000/api/v1/models
# or
curl http://127.0.0.1:5000/api/models/list

# 3) Download & load a model (one step)
curl -X POST http://127.0.0.1:5000/api/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{ "model_id": "da3nested-giant-large" }'

# 4) Two-step alternative: select then load
curl -X POST http://127.0.0.1:5000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{ "model_id": "da3nested-giant-large" }'
curl -X POST http://127.0.0.1:5000/api/load_model

# 5) Check model loading status
curl http://127.0.0.1:5000/api/model_status
# or
curl http://127.0.0.1:5000/api/v1/health

# 6) Run inference (multipart form-data)
curl -X POST http://127.0.0.1:5000/api/v1/infer \
  -F "file=@/home/robit/Pictures/33-1021508595.jpg" \
  -F "resolution=768" \
  -F "max_points=1500000"
# (defaults: resolution=504, max_points=1000000)

# 7) Poll job status (replace with the returned job_id)
curl http://127.0.0.1:5000/api/v1/jobs/job_123456789

# 8) Export point cloud (JSON)
curl http://127.0.0.1:5000/api/v1/export/glb -o export.json
```

Point cloud geometry returns `vertices` as `[x, y, z]` triples and `colors` as `[r, g, b]` (0–255 or normalized depending on path); check `metadata` for counts and resolution.

## Network access
- The Flask app already binds to `0.0.0.0` (see `app.run(host='0.0.0.0', port=<port>)`), so once the server is running you can reach it from other machines on your LAN at `http://<your-lan-ip>:<port>` (e.g., `http://192.168.1.20:5000`).
- CORS is enabled (`flask_cors.CORS(app)`), so browser-based clients on other hosts can call these endpoints if they point to your server URL.
- If you port-forward the chosen port, the same endpoints become reachable externally; add your own auth/reverse proxy in front if you need to gate access.
*** End Patch
