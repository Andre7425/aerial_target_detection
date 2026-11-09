# Aerial Target Detection for Edge Devices 🚁

An end-to-end Computer Vision pipeline optimized for real-time aerial target detection (cars, vans, trucks, buses) from UAV feeds. Designed with a focus on Defense Tech and tactical monitoring requirements, this project processes raw aerial datasets, applies environment-specific augmentations, and exports a lightweight model ready for edge deployment.

## 📌 Project Architecture

This project simulates a real-world Data Science workflow for Edge AI:
1. **Data Engineering:** Automated downloading and filtering of the VisDrone dataset to isolate specific vehicle classes, reducing noise and optimizing the model for tactical tracking.
2. **Domain-Specific Augmentation:** Integration of `Albumentations` to simulate real UAV flight conditions (Motion Blur, Gaussian Noise, Brightness fluctuations).
3. **Model Fine-Tuning:** Training `YOLO11n` (Nano) via PyTorch to achieve a balance between accuracy and computational efficiency.
4. **Edge Optimization:** Exporting the trained PyTorch graph to `ONNX` format for high-speed, framework-agnostic inference on microcontrollers (e.g., NVIDIA Jetson).

## 🚀 Performance Metrics

Tested on an NVIDIA GeForce RTX 2060 (6GB VRAM):
* **mAP50:** 0.449 (achieved in just 15 epochs)
* **Inference Speed:** 2.9ms per image
* **Throughput:** ~340 FPS (Frames Per Second), heavily exceeding the standard 30-60 FPS requirement for real-time drone cameras.

## 🛠️ Tech Stack
* **Frameworks:** PyTorch, Ultralytics (YOLO11), ONNX Runtime
* **Data & Processing:** OpenCV, Pandas, Albumentations
* **Environment:** WSL2 (Ubuntu), Python, CUDA 12.1

## 📂 Visual Results

### UAV Condition Simulation (Data Processing)
*(Add the screenshot generated in Cell 4 here, showing the original vs augmented image)*
`![Augmentation Example](docs/augmentation.jpg)`

### Inference Result
*(Add the screenshot generated in Cell 6 here, showing the bounding boxes on a test image)*
`![Inference Example](docs/inference.jpg)`

## 💻 Quick Start

### 1. Clone & Setup
```bash
git clone [https://github.com/yourusername/aerial_target_detection.git](https://github.com/yourusername/aerial_target_detection.git)
cd aerial_target_detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt