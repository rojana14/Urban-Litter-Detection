# 🗑️ Urban Litter Detection Using Mask R-CNN & YOLOv8

## 📌 Project Overview
This project is part of my **Data Science Capstone** and focuses on detecting trash in street-level images using **deep learning models**.  
The goal is to build a robust object detection pipeline and correlate **trash density** with **school violation datasets** (teacher safety and socio-economic factors).

We implemented and compared two state-of-the-art models:
- **Mask R-CNN** (segmentation + bounding box detection)
- **YOLOv8** (real-time object detection & segmentation)

---

## 📂 Dataset
- **143 custom street images** manually annotated for trash.
- **COCO-style dataset format** for compatibility with PyTorch/Detectron2.
- Dataset split: **80% training / 20% validation**.
- **Data augmentation applied**:
  - Random flips, rotations
  - Color jittering
  - Scaling and random cropping

---

## ⚙️ Methods
### Mask R-CNN
- Implemented using PyTorch & Detectron2.
- Custom COCO-style dataset loader.
- Generated both **bounding boxes** and **segmentation masks**.

### YOLOv8
- Implemented using Ultralytics YOLOv8.
- Experiments on both **detection** and **segmentation** tasks.
- Faster inference, lightweight, better mAP performance.

---

## 📊 Results

### Mask R-CNN (COCO Evaluation)
- **Bounding Box**  
  - mAP@0.5 = **0.089**  
  - Recall (AR@100) = **0.088**

- **Segmentation**  
  - mAP@0.5 = **0.101**  
  - Recall (AR@100) = **0.106**

### YOLOv8 (Ultralytics Evaluation)
- **Bounding Box**  
  - Precision = **0.654**  
  - Recall = **0.326**  
  - mAP@0.5 = **0.371**

- **Segmentation**  
  - Precision = **0.547**  
  - Recall = **0.395**  
  - mAP@0.5 = **0.402**

### Comparison Table

| Model                | mAP@0.5 | Precision | Recall | Notes |
|----------------------|---------|-----------|--------|-------|
| Mask R-CNN (BBox)    | 0.089   | 0.089*    | 0.088  | Better on large items, struggles with clutter. |
| Mask R-CNN (Segm)    | 0.101   | 0.101*    | 0.106  | Segmentation stronger than bbox, slower inference. |
| YOLOv8 (BBox)        | 0.371   | 0.654     | 0.326  | Higher precision, moderate recall, fast (~40 ms/img). |
| YOLOv8 (Segm)        | 0.402   | 0.547     | 0.395  | Best overall performance, balanced trade-off. |

\*For Mask R-CNN, AP@0.5 is used as precision proxy.

---

## 📈 Visualizations

### Training/Validation Loss for Masked R-CNN
<img src="images/Train_Validation_masked_rcnn.png" width="400"/>

### Inference Examples – Masked R-CNN
<img src="images/Masked_rcnn1.png" width="400"/>

### Inference Examples – YOLOv8
<img src="predicted_images_yolov8/527be217-trash10.jpg" width="400"/>

---

# 🚀 Phase 2 – YOLOv8-Segmentation + Pseudo-Labeling + GIS Integration

## 🎯 Project Update
In this phase, the project expanded from pure detection:
1. **YOLOv8-Segmentation (yolov8s-seg.pt)** for instance segmentation.  
2. **Semi-supervised pseudo-labeling** to auto-annotate unlabeled data.  
3. **Folium + DBSCAN** for geo-spatial hotspot visualization.  

---

## 📂 Dataset 

| Split | Images | Labels | Objects (`Trash`) |
|:--|:--:|:--:|:--:|
| **Train** | 56 | 56 | 164 |
| **Validation** | 14 | 14 | 43 |
| **Unlabeled** | 389 | – | – |

- Single class → `Trash`  
- Augmentations: mosaic, mixup, color jitter, random affine  
- All runs executed on Colab (T4 GPU)

---

## ⚙️ Model Configuration

| Parameter | Setting |
|:--|:--|
| Architecture | YOLOv8-Seg (small variant `yolov8s-seg.pt`) |
| Task | Instance Segmentation |
| Image Size | 896 × 896 px |
| Epochs | 60 |
| Batch Size | 8 |
| Optimizer | AdamW (lr = 0.01) |

---

## 🔁 Semi-Supervised Pseudo-Labeling
High-confidence predictions (`conf ≥ 0.6`) on unlabeled images were converted to YOLO labels and merged with the training set, then retrained for 60 epochs.

---

## 📊 Evaluation – Baseline vs After Pseudo-Labeling

| Metric | Baseline | After-Pseudo | Δ |
|:--|:--:|:--:|:--:|
| Precision | 0.47 | **0.51** | +0.04 |
| Recall | 0.39 | **0.54** | **+0.15** |
| mAP@50 | 0.40 | **0.46** | +0.06 |
| mAP@50–95 | 0.26 | **0.27** | +0.01 |

📈 **Result:** Recall improved by 15 %, confirming that pseudo-labeling successfully expanded the model’s detection coverage.

---

## 🧩 Figures

### Baseline Training Loss Curves
<img src="images/baseline_loss_curves.png" width="400"/>

### Baseline vs After-Pseudo Performance
<img src="images/metrics_comparison_bar.png" width="400"/>

### YOLOv8-Seg Predictions
<img src="images/detections_side_by_side.png" width="400"/>

### Loss Curve Comparison
<img src="images/after_loss_curves.png" width="400"/>

| Figure | Description |
|:--|:--|
| **1 – Baseline Training Loss Curves** | Loss convergence for initial training. | 
| **2 – Baseline vs After-Pseudo Performance** | Bar chart comparing Precision/Recall/mAP. |  
| **3 – YOLOv8-Seg Predictions** | Side-by-side detections showing mask accuracy improvement. | 
| **4 – Loss Curve Comparison** | Smoother and earlier stabilization after retraining. | 

---

## 🌍 Geo-Spatial Heatmap & Clustering

To analyze litter hotspots, a Folium + DBSCAN module was used:

```python
DBSCAN(eps=0.002, min_samples=8)
