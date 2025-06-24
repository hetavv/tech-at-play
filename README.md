# Object Detection Comparison with YOLOv8

## Folder Structure

```plaintext
assignment3/
├── Dockerfile              # Docker setup for running inference
├── inference.py            # Script for model inference and comparison
├── requirements.txt        # Python dependencies
├── images/                 # Input images (10 stock images)
│   └── *.jpg
├── outputs/                # Output files (plots, JSON, CSV)
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   ├── model_comparison_plots.png
│   ├── yolov8n_details.json
│   ├── yolov8s_details.json
│   └── ...
```

## Docker Commands

### 1. Build the Docker image

```bash
docker build -t yolo-compare .
```

### 2. Run the container with volume mounts

```bash
docker run \
  -v $(pwd)/images:/app/images \
  -v $(pwd)/outputs:/app/outputs \
  yolo-compare
```

> Make sure you're inside the `assignment3/` directory when running these commands.
