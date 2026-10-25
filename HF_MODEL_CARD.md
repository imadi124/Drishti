---
language:
- en
license: mit
library_name: keras
tags:
- satellite-imagery
- earth-observation
- remote-sensing
- semantic-segmentation
- u-net
- inceptionresnetv2
- tflite
- edge-ai
- post-training-quantization
datasets:
- humans-in-the-loop/dubai-satellite-imagery
metrics:
- mean_iou
- f1
pipeline_tag: image-segmentation
---

# InceptionResNetV2-UNet: Dubai Satellite Semantic Segmentation & Edge Optimization

This repository hosts a production-ready, edge-optimized deep learning model designed for multi-class land-cover semantic segmentation from high-resolution aerial and satellite imagery (Dubai Earth Observation Dataset).

The project explores the **accuracy vs. efficiency trade-off** for deploying aerial segmentation on resource-constrained compute platforms (e.g., on-board satellite payloads, drones, and edge CPU proxies).

---

## 📌 Model Summary

- **Architecture:** U-Net with an `InceptionResNetV2` feature extraction encoder
- **Total Parameters:** ~36.79M (Trainable: ~36.75M)
- **Input Resolution:** 512 × 512 × 3 (RGB, normalized to [0, 1])
- **Output:** 512 × 512 × 6 (Multi-channel class probability map)
- **Target Classes:** 6 categories (Water, Vegetation, Land, Road, Building, Unlabeled)
- **Optimization:** Dynamic Range Post-Training Quantization (PTQ) via TensorFlow Lite

---

## 📊 Quantization & Edge Benchmarks

We evaluated the performance and resource trade-offs between standard floating-point precision (FP32) and dynamic 8-bit integer quantization (INT8) on an edge-proxy CPU architecture:

| Format | Precision | Model Size (MB) | Size Reduction | Overall mIoU | 5-Class Terrain mIoU | CPU Latency / Image |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras (.keras)** | FP32 | 423.14 MB | Baseline (1.0x) | **59.09%** | **70.34%** | 4,693 ms |
| **TFLite** | FP32 | 147.18 MB | ~2.9x smaller | **59.09%** | **70.34%** | ~4,200 ms |
| **TFLite (Dynamic)**| INT8 | **35.89 MB** | **11.8x smaller (-91.5%)** | **51.08%** | **60.28%** | ~3,100 ms |

> **Key Takeaway:** Dynamic INT8 quantization reduces storage requirements by **91.5%** (saving **387.25 MB** of memory) while maintaining **60.28%** terrain mIoU, making it viable for on-orbit and edge deployment.

### 🔍 Side-by-Side Class Breakdown: Baseline (FP32) vs. Quantized (INT8)

| Class | Baseline IoU (FP32) | INT8 IoU (Quantized) | Change ($\Delta$ IoU) | Status |
| :--- | :---: | :---: | :---: | :--- |
| 🌊 **Water** | 0.9015 | 0.8849 | -0.0166 (-1.7%) | 🟢 Preserved almost perfectly |
| 🌿 **Vegetation** | 0.6163 | 0.6190 | +0.0027 (+0.3%) | 🟢 Completely preserved |
| 🏜️ **Land** | 0.8083 | 0.7761 | -0.0322 (-3.2%) | 🟢 Very stable |
| 🛣️ **Road** | 0.5105 | 0.3327 | -0.1778 (-17.8%) | 🟡 Moderate drop |
| 🏢 **Building** | 0.6806 | 0.4012 | -0.2794 (-27.9%) | 🔴 Major drop |
| ❓ **Unlabeled** | 0.0282 | 0.0509 | +0.0227 | — |
| **Overall mIoU** | **59.09%** | **51.08%** | **-8.01%** | |
| **5-Class Terrain mIoU** | **70.34%** | **60.28%** | **-10.06%** | |
| **Model Size** | **423.14 MB** | **35.89 MB** | **-387.25 MB** | 🚀 **11.8x Smaller** |

---

## 🎨 Class & Color Ontology

The model outputs predictions across 6 land-cover classes based on the MBRSC Dubai Satellite Dataset:

| Class | R | G | B | Color Preview |
| :--- | :---: | :---: | :---: | :--- |
| **Water** | 226 | 169 | 41 | `#E2A929` |
| **Land (unpaved)** | 132 | 41 | 246 | `#8429F6` |
| **Road** | 110 | 193 | 228 | `#6EC1E4` |
| **Building** | 60 | 16 | 152 | `#3C1098` |
| **Vegetation** | 254 | 221 | 58 | `#FEDD3A` |
| **Unlabeled** | 155 | 155 | 155 | `#9B9B9B` |

---

## 💻 Quickstart Inference Guide

### 1. Load Full Precision Model (Keras / TensorFlow)
```python
import tensorflow as tf
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/dubai-satellite-segmentation-inceptionresnetv2-unet",
    filename="InceptionResNetV2-UNet.keras"
)

model = tf.keras.models.load_model(model_path, compile=False)
```

### 2. Edge / CPU Inference with Quantized TFLite
```python
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Download quantized INT8 model (only 35.89 MB)
tflite_path = hf_hub_download(
    repo_id="your-username/dubai-satellite-segmentation-inceptionresnetv2-unet",
    filename="InceptionResNetV2-UNet_int8.tflite"
)

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pass normalized image (1, 512, 512, 3)
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()

# Output shape: (1, 512, 512, 6)
predictions = interpreter.get_tensor(output_details[0]['index'])
mask = np.argmax(predictions[0], axis=-1)
```

---

## ⚠️ Operational Limitations & Edge Cases

1. **Spatial Resolution:** Calibrated for high-resolution aerial and satellite tiles (~0.5m – 1.0m Ground Sample Distance). Degrades on lower-resolution public sensors like Sentinel-2 (10m).
2. **Dense Occlusion:** Extreme sun glint, heavy maritime haze, or high-rise building shadows can cause confusion between asphalt roads and deep water.
3. **Quantization Degradation:** Dynamic INT8 preserves large spatial topologies (water, land, vegetation) with minimal loss, but fine-detail geometric edges (roads and building footprints) suffer under post-training dynamic quantization without quantization-aware training (QAT).
