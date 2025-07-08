
# EventSem: Event-Aware Multi-Source Semantic Enhancement for Video Moment Retrieval


## 🎯 Overview

EventSem addresses two fundamental limitations in existing video moment retrieval (VMR) methods:
- **Semantic representation inadequacy** due to reliance on single-source embeddings
- **Temporal structure ignorance** by treating videos as uniform sequences

Our framework introduces two plug-and-play modules that can enhance any proposal-based VMR method:

1. **Multi-Source Semantic Enhancement (MSSE)**: Bridges semantic gaps through adaptive fusion of external linguistic knowledge
2. **Event-Aware Temporal Grounding (EATG)**: Leverages natural video structure for intelligent proposal generation

## 🚀 Key Features

- **🔥 State-of-the-art Performance**: Achieves new SOTA on TACoS, Charades-STA, and QVHighlights
- **🔧 Plug-and-Play**: Seamlessly integrates into any proposal-based VMR architecture
- **💪 Robust**: Superior performance on linguistically diverse queries


## 🛠️ Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.1
- CUDA >= 11.0

### Setup
```bash
# Clone the repository
cd EventSem

# Create conda environment
conda create -n eventsem python=3.8
conda activate eventsem

# Install dependencies
pip install -r requirements.txt

```

## 📁 Data Preparation

### Download Datasets
```bash
Download QVHighligths, Charades-STA, and TACoS features, please follow th instruction of [CG-DETR](https://github.com/wjun0830/CGDETR/)
```

### Linguistically Diverse Test Sets
Our constructed test sets with semantically equivalent query reformulations:
```bash
# Download our semantic robustness evaluation sets
# to do 
```

## 🔧 Training

### QVHighlights

```python
bash EventSem/scripts/train_qv_slowclip.sh
```

### Charades-STA

For VGG feature:

```python
bash EventSem/scripts/charades-sta/train_vgg.sh
```

For SlowFast+Clip feature:
```python
bash EventSem/scripts/charades-sta/train.sh
```

### TACoS

```python
bash EventSem/scripts/TACoS/train.sh
```


## 📋 Evaluation

### Standard Evaluation
```bash
# Evaluate on TACoS
bash Event/scripts/inference.sh data/MR.py results/TACoS/model_best.ckpt 'val'

# Evaluate on all datasets
bash scripts/eval_all_datasets.sh
```

### Semantic Robustness Evaluation
```bash
# Evaluate on linguistically diverse test sets
replace the original test file to linguistically diverse test sets
```



## 📈 Model Zoo
We provide multiple checkpoints and training logs here. Configuration can be find in each opt.json file.
|Datasets |Model file|
|-------|-------|
|QVHighlights| -|
|Charades(VGG)| -|
|Charades(SF+C)| -|
|TACoS| -|




## 🙏 Acknowledgments

- Built upon [QD-DETR](https://github.com/wjun0830/QD-DETR) and [TR-DETR](https://github.com/microsoft/VideoX)
- Thanks to the creators of TACoS, Charades-STA, and QVHighlights datasets
