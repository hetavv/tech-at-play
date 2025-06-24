import os
import time
import json
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi


IMAGE_DIR = "images"
OUTPUT_DIR = "output"
MODELS = {"yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

results_summary = []

for model_name, model_path in MODELS.items():
    model = YOLO(model_path)
    model_outputs = []
    total_time = 0
    all_classes = []

    for image_file in image_files:
        img_path = os.path.join(IMAGE_DIR, image_file)
        img = Image.open(img_path)

        start = time.time()
        results = model.predict(source=img, save=False, verbose=False)
        elapsed = time.time() - start
        total_time += elapsed

        result = results[0]
        boxes = result.boxes
        classes = result.names

        class_counts = defaultdict(int)
        for cls_id in boxes.cls.tolist():
            class_name = classes[int(cls_id)]
            class_counts[class_name] += 1
            all_classes.append(class_name)

        # Save annotated image
        img_save_path = os.path.join(OUTPUT_DIR, f"{model_name}_{image_file}")
        result.save(filename=img_save_path)

        model_outputs.append(
            {
                "image": image_file,
                "detections": sum(class_counts.values()),
                "unique_classes": len(class_counts),
                "class_counts": dict(class_counts),
                "inference_time": round(elapsed, 4),
            }
        )

    avg_time = round(total_time / len(image_files), 4)
    total_detections = sum(m["detections"] for m in model_outputs)
    diversity = len(set(all_classes))

    results_summary.append(
        {
            "model": model_name,
            "avg_inference_time": avg_time,
            "total_detections": total_detections,
            "class_diversity": diversity,
        }
    )

    # Save per-image results
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_details.json"), "w") as f:
        json.dump(model_outputs, f, indent=2)

# Save comparison table
df = pd.DataFrame(results_summary)
df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
df.to_json(
    os.path.join(OUTPUT_DIR, "model_comparison.json"), orient="records", indent=2
)


# save plots as well
df = pd.read_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

# Plot 1: Horizontal Grouped Bar (Detections & Class Diversity)
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
sns.set_style("whitegrid")

# Total Detections & Class Diversity
bar_data = df.set_index("model")[["total_detections", "class_diversity"]]
bar_data.plot(kind="barh", ax=axs[0], color=["#4C72B0", "#55A868"])
axs[0].set_title("Detections vs Class Diversity")
axs[0].set_xlabel("Count")
axs[0].legend(loc="lower right")

# Plot 2: Inference Time Bar
sns.barplot(
    x="model",
    y="avg_inference_time",
    hue="model",
    data=df,
    ax=axs[1],
    palette="Set2",
    legend=False,
)
axs[1].set_title("Average Inference Time (s)")
axs[1].set_ylabel("Seconds")
axs[1].set_xlabel("")
for i, row in df.iterrows():
    axs[1].text(
        i, row.avg_inference_time + 0.01, f"{row.avg_inference_time:.2f}", ha="center"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_plots.png"))
