# Distributed Object Detection with DDP, FSDP, and DeepSpeed ZeRO

## 📌 Overview

This project evaluates and compares the performance of three distributed training strategies in PyTorch for object detection using the Faster R-CNN model:
- **DistributedDataParallel (DDP)**
- **Fully Sharded Data Parallel (FSDP)**
- **DeepSpeed ZeRO Stage 1**

The goal is to study how these methods differ in training time, memory consumption, and accuracy (mean Average Precision, mAP) under 2-GPU and 4-GPU setups.

---

## 🔍 Problem Statement

Object detection is a key task in computer vision. However, training detection models like Faster R-CNN at scale is compute-intensive. This project investigates how to accelerate training with minimal accuracy loss using state-of-the-art parallel training approaches.

---

## 🧠 Methods Evaluated

1. **DDP** – Default PyTorch method for data parallelism using gradient synchronization.
2. **FSDP** – A memory-efficient alternative that shards model weights and optimizer states.
3. **ZeRO (Stage 1)** – DeepSpeed method that shards optimizer states for large model scaling.

---

## 📊 Dataset

- **COCO 2017** subset
  - 12,000 training images
  - 2,000 validation images
- Applied basic `ToTensor()` transforms

---

## ⚙️ Framework and Platform

- PyTorch 2.x
- DeepSpeed for ZeRO
- NCCL backend
- Lambda Labs instance with 4× NVIDIA A6000 GPUs

---

## 📈 Performance Metrics

- **Training time per epoch**
- **Average loss**
- **mAP (mean Average Precision)**
- **Scalability (2-GPU vs. 4-GPU speedup)**

---

## 🧪 Results Summary

| Method | 4-GPU Speedup | Memory Efficiency | mAP | Integration Effort |
|--------|----------------|--------------------|-----|----------------------|
| DDP    | ~1.6×         | ❌ High usage      | ✅ Best    | ✅ Easy            |
| FSDP   | ~1.55×        | ✅ Very efficient  | ~    | ❌ Medium effort   |
| ZeRO   | ~1.7×         | ✅ Excellent       | ✅ Stable | ⚠️ Extra dependency |

---

## 🔮 Future Work

- Evaluate ZeRO Stage 2 and 3
- Apply mixed-precision (AMP)
- Test on full COCO (118k images)
- Scale to 8+ GPU clusters
- Try with Mask R-CNN or Swin Transformer

---

## 📎 Downloads

- 📄 [Final Project Report (PDF)](link-to-your-report)
- 📊 [Slides (PPT)](link-to-your-slides)

---

## 🔗 References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [COCO Dataset](https://cocodataset.org/#home)
- [PyTorch DDP Docs](https://pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch FSDP Docs](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO](https://www.microsoft.com/en-us/research/blog/zero-optimizations-for-training-large-deep-learning-models/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)

---

## 📧 Contact

**Biplove Lamsal**  
University of Texas at San Antonio  
Email: biplove.lamsal@example.edu
