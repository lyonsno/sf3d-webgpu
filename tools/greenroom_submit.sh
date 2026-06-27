#!/usr/bin/env bash
# Submit an SF3D WebGPU inference smoke to the GPU greenroom.
# Usage: tools/greenroom_submit.sh [image_path]
#
# Writes job request to greenroom pending queue.
# Returns job_id for status polling.

set -euo pipefail

GREENROOM_DIR="${GPU_GREENROOM_DIR:-$HOME/.local/state/gpu-greenroom}"
IMAGE="${1:-$HOME/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png}"
JOB_ID=$(python3 -c "import hashlib, time; print(hashlib.md5(str(time.time()).encode()).hexdigest()[:12])")
OUTPUT_DIR="$GREENROOM_DIR/outputs/$JOB_ID"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$GREENROOM_DIR/pending/$JOB_ID"

cat > "$GREENROOM_DIR/pending/$JOB_ID/request.json" << EOF
{
    "job_type": "sf3d_webgpu",
    "input_path": "$IMAGE",
    "output_dir": "$OUTPUT_DIR",
    "params": {},
    "job_id": "$JOB_ID",
    "submitted_at": $(python3 -c "import time; print(time.time())")
}
EOF

echo "Submitted sf3d_webgpu job: $JOB_ID"
echo "Image: $IMAGE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Poll status: cat $GREENROOM_DIR/running/$JOB_ID/status.json 2>/dev/null || cat $GREENROOM_DIR/done/$JOB_ID/status.json 2>/dev/null || echo pending"
echo "View report: cat $OUTPUT_DIR/report.txt"
