# Research Proposal — Paper (Efficient / Edge Remote-Sensing Segmentation)

## Working title (pick one later)
- *"Small Enough to Fly: Compressing Semantic Segmentation Models for On-Board Remote Sensing"*
- *"How Small Can You Go? Accuracy vs. Efficiency Trade-offs for Edge Deployment of Aerial Image Segmentation"*

## One-line summary
We take a semantic-segmentation model for satellite/aerial imagery and apply different architectures and measure the metrics shrink it with quantization, pruning, and knowledge distillation, then map out exactly how much accuracy (mIoU) and other metrics you trade for how much smaller/faster/cheaper the model becomes — so it can run *on the drone or satellite itself*, not just on a ground GPU.

---

## 1. Problem statement
Segmenting aerial and satellite images (labelling every pixel as water, building, road, vegetation, etc.) is normally done on powerful ground GPUs. But there is a growing need to run it **on-board** — on the drone or satellite as it flies — to (a) avoid transmitting huge raw images over slow/expensive links, (b) react instantly (disaster response, precision agriculture, defense), and (c) work with no connectivity. The problem: on-board chips are small, low-power, and memory-limited, and a standard segmentation network (a full U-Net or a transformer) is **too big, too slow, and too power-hungry** to run there.

This paper asks: **how far can we compress a remote-sensing segmentation model before its quality breaks — and which compression strategy gives the best accuracy for the least size, latency, and compute?**

## 2. The gap it solves (positioned honestly)
Model compression (quantization, pruning, distillation) is well studied, and there are efficient segmentation networks (e.g., ENet, ICNet, BiSeNet, Fast-SCNN, MobileNet/EfficientNet backbones). So "compress a model" is **not** novel on its own — do not pitch it that way.

The **specific, defensible gaps** (the combination is the contribution):
1. **Remote-sensing segmentation specifically.** RS images have heavy **class imbalance** (tiny/rare classes), high resolution, and patch-based processing. How compression interacts with these RS-specific properties is under-studied compared to natural-image classification.
2. **Systematic comparison of *combinations*.** Most papers study one technique. We compare quantization vs pruning vs distillation **and their combinations** on one clean accuracy-vs-cost frontier.
3. **Real deployment metrics, not just theory.** We report measured **latency, memory, and throughput** on a constrained device (or an explicit CPU/edge proxy) plus **FLOPs** — not only parameter counts.
4. **Class-level degradation analysis** (your differentiator). Using your class-wise IoU + error-heatmap skills, we show *which land-cover classes fail first* under compression (e.g., do rare or fine-detail classes collapse while common ones survive?). That's genuinely useful and rarely reported.

> ⚠️ **Do a related-work search before writing** (the live search tool was unavailable at drafting time). Search for: "efficient semantic segmentation remote sensing", "model compression segmentation edge", "quantization pruning knowledge distillation segmentation", "lightweight U-Net satellite", "on-board / on-orbit inference remote sensing". Position against the efficient-segmentation networks above. **No citations are invented here** — supply verified ones.

## 3. Research questions
- **RQ1.** How much can we compress an RS segmentation model (via quantization, pruning, distillation, and combos) before mIoU drops unacceptably?
- **RQ2.** Which method/combination gives the best **accuracy-vs-cost** frontier (mIoU vs size / latency / FLOPs)?
- **RQ3.** *Which classes and features degrade first* under compression? (class-wise IoU, confusion matrices, error heatmaps)
- **RQ4.** What is the **real on-device performance** (latency, memory, throughput) on constrained hardware or a clearly-stated edge proxy?

## 4. Approach (step by step, plain language)
1. **Baselines.** Reproduce/strengthen your custom U-Net on the dataset, and add 1–2 stronger-but-still-light baselines (e.g., a MobileNetV3- or EfficientNet-backbone U-Net, or a lightweight net like BiSeNet / Fast-SCNN) so results aren't tied to one architecture.

For research your notebook, the easiest architecture swaps with least code changes are that I am going to apply are:

EfficientNet-B0 + U-Net
MobileNetV2 + U-Net
ResNet50 + U-Net

##### An important part in the approch is using optuna and bayesian search optimation for hyperparameter tuning 
learning rate, batch size, optimizer type, number of filters, dropout rate, augmentation strength, target image size, number of epochs, whether to use pretrained backbone, activation choices, loss weighting between classes, Also, we can use the different architectues also to find out the best architectures using this optuna

2. **Apply the three compression tools:**
   - **Quantization** — convert 32-bit weights to 8-bit (INT8). Use **post-training quantization (PTQ)** first (easiest), then **quantization-aware training (QAT)** if time. (Your existing quantization intuition transfers directly.)
   - **Pruning** — remove low-importance weights/filters (magnitude-based, structured where possible so it actually speeds up on hardware). Sweep several pruning ratios (e.g., 30/50/70%).
   - **Knowledge distillation** — train a small "student" network to mimic a larger "teacher" (match its output probabilities and/or feature maps).
   - **Combinations** — prune+quantize, distill+quantize.
3. **Measure accuracy** (mIoU, per-class IoU, pixel accuracy) on the test set for every variant.
4. **Measure efficiency** — parameters, model size (MB), FLOPs/MACs, inference latency (ms/image), throughput (FPS), peak memory — on a desktop CPU as an edge proxy and, if you can get one, a real edge device (Raspberry Pi 4 / Jetson Nano).
5. **Degradation analysis** — compare per-class IoU before/after, confusion matrices, and pixel-level error heatmaps to see *what* breaks.
6. **Plot the Pareto frontier** (accuracy vs each cost axis) and give a "use X to hit Y constraint" recommendation.

## 5. Models & compression stack (technologies)
- **Framework:** TensorFlow / Keras (your stack) + **TensorFlow Model Optimization Toolkit** (pruning, QAT) + **TensorFlow Lite** (INT8 conversion and on-device latency benchmarking — ideal for the edge story). (PyTorch + ONNX Runtime is an alternative, but stay in TF.)
- **Architectures:** your custom U-Net (baseline); a MobileNetV3/EfficientNet-backbone U-Net; optionally BiSeNet / Fast-SCNN as a "designed-for-speed" reference.
- **Compression:** PTQ + QAT (INT8); magnitude/structured pruning; response- and feature-based knowledge distillation.
- **Measurement:** `keras-flops` or a profiler for FLOPs/MACs; TFLite Benchmark tool for latency; track model size and peak RAM.
- **Data pipeline:** your existing **Patchify** 256×256 patch pipeline, OpenCV, scaling/normalization.
- **Analysis:** scikit-learn (IoU, confusion matrix), matplotlib (frontier plots, heatmaps).
- **Edge hardware (optional but strong):** Raspberry Pi 4, Jetson Nano, or Google Coral. If unavailable, use **single-thread desktop CPU latency + FLOPs** as an explicit proxy and say so.
- **Compute:** Colab Pro / A100 — training these small models is fast; compression and evaluation are cheap.

## 6. Datasets (confirm every ID/source before use)
**Primary**
- **MBRSC Dubai aerial dataset** — your existing 6-class set (Kaggle: "Semantic Segmentation of Aerial Imagery", by Humans in the Loop / MBRSC). Reuse your pipeline. *Verify current link/license.*

**Add for generalization (pick 1–2)**
- **LoveDA** — land-cover, urban+rural, domain-shift friendly. HF: try `Junjue-Wang/LoveDA`. *Verify.*
- **ISPRS Potsdam / Vaihingen** — classic 2D semantic labeling benchmarks (registration may be required). *Verify.*
- **DeepGlobe Land Cover** or **UAVid** (drone video frames) — good "edge/drone" fit. *Verify.*

> Recommended minimal set for a clean 3–4 month paper: **your MBRSC Dubai set + one public benchmark (LoveDA or Potsdam).** That's enough to show the findings generalize.

## 7. Metrics
- **Accuracy:** mIoU, per-class IoU, pixel accuracy, Dice/F1.
- **Other Important Metrics to report are :** Dice, IoU / mIoU, per-class IoU, precision / recall / F1, pixel accuracy, confusion matrix and class-wise metrics
- **Efficiency:** #parameters, model size (MB), FLOPs/MACs, latency (ms/image, CPU + edge), throughput (FPS), peak RAM.
- **Trade-off:** compression ratio, speedup, and the accuracy-vs-{size, latency, FLOPs} **Pareto frontier**.
- **Degradation:** per-class IoU drop, confusion-matrix shifts, error heatmaps.

## 8. Experiment matrix
`{U-Net, 1–2 light architectures} × {none, PTQ-INT8, QAT, prune@{30,50,70%}, distillation, prune+quant, distill+quant} × {MBRSC, one public set}`
→ each cell yields accuracy + efficiency + per-class degradation. **Start minimal:** your U-Net on MBRSC — baseline vs INT8 vs one pruning level vs distillation — then expand.

## 9. Expected contributions
1. A **systematic accuracy-vs-efficiency study** of quantization/pruning/distillation (and combos) for remote-sensing semantic segmentation, with **measured deployment metrics**.
2. Practical guidance on **which compression strategy best preserves segmentation quality** per unit of size/latency/compute.
3. A **class-level degradation analysis** showing which land-cover classes suffer most under compression.
4. A reproducible pipeline **and a deployable lightweight model** for on-board aerial mapping.

## 10. Compute feasibility (fits 3–4 months)
Your baseline is tiny (1.94M params) and trains fast; even larger-backbone baselines are quick on an A100. Compression and evaluation are cheap. The main setup effort is the pruning/quantization toolkits and the **measurement harness** (FLOPs + latency). All well within your compute and timeline.

## 11. Risks & mitigations
- **No physical edge device.** → Report single-thread CPU latency + FLOPs as an explicit proxy; optionally borrow a Raspberry Pi/Jetson (cheap) for one real-hardware table.
- **"Not novel enough."** → Lead with the RS-specific **class-degradation analysis** + method **combinations** + real deployment metrics, not "we quantized a model." Nail related work.
- **Baseline mIoU is modest (0.62).** → Optionally strengthen the baseline first, but frame the paper around the **trade-off**, not chasing state-of-the-art accuracy — so the contribution doesn't depend on a high number.
- **Toolkits are finicky.** → Start with TFLite PTQ (easiest win), then add pruning, then distillation, incrementally.
- **Scope creep.** → Ship the minimal matrix first; add architectures/datasets only if time allows.


## 12. Important things to include in the research paper to be called valid other than this notebook
A publication-quality study usually needs more than just one architecture and one metric.

You would normally want:

clear dataset description, train/val/test split, per-class metrics, mIoU, Dice, precision/recall,, confusion matrix, comparisons against baselines, ablation studies, reproducible config, code and seeds, hardware details, model efficiency (fps, params, memory)

#### Also for research, the result should be compared to:
- baseline U-Net
- - PSPNet
- DeepLabV3+
- SegFormer
- MobileNet/efficient backbones
- lightweight transfomer based segmentation models if deployment is the goal
- So this notebook is a very good reference model, but by itself it is not a full publication-ready benchmark.

#### If you want a model that can run on drones or satellites, you also need to care about:

- model size
- memory usage
- inference time
- FPS
- power consumption
- edge hardware compatibility

#### What would make this more publishable
If you build on this notebook, a stronger research workflow would be:
1. Baseline model
- current U-Net + InceptionResNetV2
2. Compare with several models

3. measure the metrics for research and production and deployemnt 

4. Apply compression techniques 

5. Evaluate on edge hardware

- Jetson
- Raspberry Pi
- drone onboard CPU/GPU
- satellite-like low-power environment

I don't have these edge hardware devices so I will use the below techniques to masure the acceptable metrics :

### A. CPU-only latency benchmarking on your laptop
This is the most practical option.

Measure:

inference time per image
throughput (FPS)
memory footprint
model size
parameter count
FLOPs / MACs
This proves the model is lightweight enough to be considered for edge deployment.

### B. A constrained CPU profile
Use a “proxy” configuration:

single-thread CPU mode
lower memory limit
reduced batch size
power-limited settings if possible
This imitates a resource-constrained environment more realistically than standard desktop inference.


### C. TensorFlow Lite / ONNX Runtime benchmarking
If you use TensorFlow/Keras, this is very good.

You can benchmark:

original model
INT8 quantized model
pruned model
distilled student model
with:

TFLite benchmark tool
ONNX Runtime
XNNPACK / CPU delegate configuration
This is highly credible in a paper because it  it is a real inference benchmark, not just a theoretical estimate.


### D. FLOPs, parameters, memory, and storage analysis
These are accepted as deployment evidence when hardware is not available.

Examples:

model size before vs after compression
parameters reduced by X%
FLOPs reduced by Y%
peak memory use
latency reduction after pruning/quantization
This gives a clear “efficiency frontier.”



### You should say something like this in the research papers:

We evaluate the proposed compressed segmentation models on a constrained CPU/edge proxy rather than the final embedded target platform, because the target hardware was unavailable during the study. We report latency, model size, memory consumption, and complexity metrics to estimate deployment feasibility and provide a realistic accuracy-efficiency trade-off.

This is honest and academically acceptable.