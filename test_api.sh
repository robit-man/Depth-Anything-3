#!/bin/bash
# Test script for Depth Anything 3 RESTful API
# Demonstrates curl-based API usage

API_BASE="http://localhost:5000"

echo "=========================================="
echo "Depth Anything 3 - API Test Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
curl -s "$API_BASE/api/v1/health" | json_pp
echo ""
echo ""

# Test 2: Process image from URL
echo -e "${YELLOW}Test 2: Process image from URL${NC}"
curl -s -X POST "$API_BASE/api/v1/process/image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://picsum.photos/800/600",
    "resolution": 504,
    "max_points": 100000,
    "export_format": "json"
  }' | json_pp | head -50
echo "... (truncated)"
echo ""
echo ""

# Test 3: Process local image (if provided)
if [ -n "$1" ]; then
    echo -e "${YELLOW}Test 3: Process local image (base64)${NC}"
    IMAGE_PATH="$1"

    if [ -f "$IMAGE_PATH" ]; then
        # Read and encode image to base64
        IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")
        IMAGE_EXT="${IMAGE_PATH##*.}"

        # Determine MIME type
        case "$IMAGE_EXT" in
            jpg|jpeg) MIME="image/jpeg" ;;
            png) MIME="image/png" ;;
            webp) MIME="image/webp" ;;
            *) MIME="image/jpeg" ;;
        esac

        echo "Processing: $IMAGE_PATH"
        curl -s -X POST "$API_BASE/api/v1/process/image" \
          -H "Content-Type: application/json" \
          -d "{
            \"image\": \"data:$MIME;base64,$IMAGE_BASE64\",
            \"resolution\": 504,
            \"max_points\": 500000,
            \"export_format\": \"json\",
            \"include_cameras\": true
          }" | json_pp | head -50
        echo "... (truncated)"
        echo ""
    else
        echo -e "${RED}Error: File not found: $IMAGE_PATH${NC}"
    fi
    echo ""
fi

# Test 4: Download GLB (if image provided)
if [ -n "$1" ] && [ -f "$1" ]; then
    echo -e "${YELLOW}Test 4: Download GLB${NC}"
    IMAGE_PATH="$1"
    IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")
    IMAGE_EXT="${IMAGE_PATH##*.}"

    case "$IMAGE_EXT" in
        jpg|jpeg) MIME="image/jpeg" ;;
        png) MIME="image/png" ;;
        webp) MIME="image/webp" ;;
        *) MIME="image/jpeg" ;;
    esac

    OUTPUT_GLB="test_output.glb"

    echo "Downloading GLB to: $OUTPUT_GLB"
    curl -s -X POST "$API_BASE/api/v1/process/image" \
      -H "Content-Type: application/json" \
      -d "{
        \"image\": \"data:$MIME;base64,$IMAGE_BASE64\",
        \"resolution\": 504,
        \"export_format\": \"glb\"
      }" \
      -o "$OUTPUT_GLB"

    if [ -f "$OUTPUT_GLB" ]; then
        FILE_SIZE=$(ls -lh "$OUTPUT_GLB" | awk '{print $5}')
        echo -e "${GREEN}✓ GLB saved: $OUTPUT_GLB ($FILE_SIZE)${NC}"
    else
        echo -e "${RED}✗ Failed to download GLB${NC}"
    fi
    echo ""
fi

# Test 5: Check model status
echo -e "${YELLOW}Test 5: Check model status${NC}"
curl -s "$API_BASE/api/model_status" | json_pp
echo ""
echo ""

echo "=========================================="
echo "API Tests Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  ./test_api.sh              # Run basic tests"
echo "  ./test_api.sh image.jpg    # Run tests with local image"
echo ""
echo "Requirements:"
echo "  - Flask server running (python3 main.py)"
echo "  - json_pp for JSON formatting (usually installed)"
echo "  - curl"
echo ""
