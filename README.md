# 🧠 Multimodal AI Inference Pipeline

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/) 
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue)]([https://github.com/Adyasha-Nanda/AI-multimodal-pipeline]))

---

## 🚀 Project Overview

This project implements a **lightweight multimodal AI inference pipeline** that combines **object detection** and **OCR** in real-time.  
The pipeline is designed to be optimised for performance, and easy to deploy.

**Core Features:**
- Input: Image file or live webcam feed
- Object Detection: YOLOv8-nano (fast and lightweight)
- OCR (Text Recognition): EasyOCR (robust for multiple fonts)


**Pipeline Flow:**
[Input Image/Webcam Frame]->[YOLOv8 Object Detector]->[Crop each detected object]->[OCR]->[Overlay boxes + text on frame]->[Display or Save Frame]
