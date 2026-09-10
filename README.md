# H-GAT: Hyperedge-Aware Graph Attention Network for Coordinated QCA Placement and Clocking

Official implementation of the paper **"Hyperedge-Aware Graph Attention Network for Coordinated QCA Placement and Clocking"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

---

## Overview

we are the first to introduce a **Graph Attention Network (GAT)** into QCA placement automation, and propose a **Hyperedge-Aware GAT model (H-GAT)**. The model encodes multi-dimensional physical information, such as spatial positions and clock phases, into graph embeddings, allowing the network to perceive placement and timing information directly at the input stage. On this basis, hyperedge attention captures group-wise constraints, and multi-head GAT layers aggregate the coupled spatial-temporal information, enabling the network to **jointly predict placement and timing**.

H-GAT successfully captures the intrinsic laws of spatial placement and timing propagation, providing a direct and accurate prediction tool for QCA automation. More importantly, this graph-learning-based feature extraction framework can serve as a fundamental support, effectively replacing traditional heuristic rules and providing reliable initial states and search foundations for subsequent complex multi-objective optimization tasks such as reinforcement learning.

---

## Key Features

- **Hyperedge-aware attention** for group-wise QCA constraints
- **Joint prediction** of block coordinates, offsets, and clock phases
- **Spatial-temporal representation learning** for QCA automation

- 24-dimensional feature components with inseparable collaborative support
- Learned feature space consistent with the unidirectional propagation of QCA clock timing
- Provides a learned prior for downstream reinforcement learning and multi-objective optimization

---

## Tasks

The model jointly predicts five targets:

| Task | Description |
|---|---|
| Block X | X coordinate of the block |
| Block Y | Y coordinate of the block |
| Offset X | X offset |
| Offset Y | Y offset |
| Clock | Clock phase |

---
