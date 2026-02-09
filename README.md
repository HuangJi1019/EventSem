
# EventSem: Plug-and-Play Multi-Source Semantic Enhancement and Event-Aware Temporal Grounding for Video Moment Retrieval


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
- **💪 Datasets and Robust**: we introduce a new SRE dataset. Superior performance on this SRE dataset.


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
Download QVHighligths, Charades-STA, and TACoS features, please follow the instruction of CG-DETR
```
### Linguistic Knowledge Datasets
```bash
run code linguistic_knowledge.py to generate the linguistic knowledge
```
The datasets file structure would be:
```
--datasets
    --charades_sta
        --clip_features
        --clip_text_features
        --slowfast_features
    --qvhighlight
        --clip_features
        --clip_text_features
        --slowfast_features
    --tacos
        --clip_features
        --clip_text_features
        --slowfast_features
    --semantic_embeddings 
        --charades-sta-token-level
        --qv_highlight_token_level
        --tacos-token-level
```

### Semantic Robustness Evaluation test sets
We rewrite original queries using T5-based paraphrasers with semantic similarity filtering (cosine similarity $\geq$ 0.85) to preserve meaning. We call this the SRE dataset. We report the Charades-STA-SRE and the TACoS-SRE datasets.

For the Charades-STA-SRE, you can find it in
```bash
data/charades_sta/charades_sta_SRE_test_tvr_format.jsonl 
```
For the TACoS-SRE, you can find it in
```bash
data/tacos/test_SRE.jsonl 
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
```

### Semantic Robustness Evaluation
```bash
# Evaluate on linguistically diverse test sets
replace the original test file to Semantic Robustness Evaluation test sets
```



## 📈 Model Zoo
We will provide it after the paper is accepted.

## 🙏 Acknowledgments

- Built upon QD-DETR,TR-DETR, and FlashVTG
- Thanks to the creators of TACoS, Charades-STA, and QVHighlights datasets
