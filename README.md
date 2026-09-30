
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
git clone https://github.com/HuangJi1019/EventSem.git
cd EventSem

conda create -n eventsem python=3.8
conda activate eventsem

pip install -r requirements.txt
```

All commands below are run from the repository root.

## 📁 Data Preparation

### Video and text features
Download the QVHighlights, Charades-STA and TACoS features following the instructions of
[CG-DETR](https://github.com/wjun0830/CGDETR) and place them under `datasets/`.

### Linguistic knowledge (MSSE features)
Download the GloVe vectors [`glove.6B.300d.txt`](https://nlp.stanford.edu/projects/glove/) into
the repository root, then generate the token-level semantic features:
```bash
python linguistic_knowledge_tacos_v2.py      # -> datasets/semantic_embeddings/tacos-token-level-v2
python linguistic_knowledge_charades_v2.py   # -> datasets/semantic_embeddings/charades-sta-token-level-v2
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
        --charades-sta-token-level-v2
        --tacos-token-level-v2
```

### Semantic Robustness Evaluation (SRE) test sets
We rewrite original queries using T5-based paraphrasers with semantic similarity filtering (cosine similarity $\geq$ 0.85) to preserve meaning. We call this the SRE dataset. We report the Charades-STA-SRE and the TACoS-SRE datasets:

| Dataset | Query file |
|---|---|
| Charades-STA-SRE | `data/charades_sta/charades_sta_SRE_test_tvr_format.jsonl` |
| TACoS-SRE | `data/tacos/test_SRE.jsonl` |

## 🔧 Training

### TACoS
```bash
bash EventSem/scripts/tacos/train.sh
```

### Charades-STA
SlowFast + CLIP features:
```bash
bash EventSem/scripts/charades_sta/train.sh
```
VGG features:
```bash
bash EventSem/scripts/charades_sta/train_vgg.sh
```

### QVHighlights
```bash
bash EventSem/scripts/train_qv_slowclip.sh
```

Extra arguments are passed through to `EventSem/train.py`, e.g. `--seed 2027` or `--exp_id my_run`.
Results are written to `results_<dataset>/<dataset>-video_tef-<exp_id>-<timestamp>/`, including
`model_best.ckpt` and `best_*_metrics.json`. Pass `--no_semantic_enhancement` to disable MSSE and
`--no_event_prior_filtering` to disable EATG.

## 📋 Evaluation

### Standard evaluation
```bash
bash EventSem/scripts/inference.sh <path/to/model_best.ckpt> data/tacos/test.jsonl
```
The model settings are read from the `opt.json` saved next to the checkpoint. Add
`--eval_results_dir <dir>` to write the predictions somewhere other than the checkpoint's directory.

### Semantic Robustness Evaluation
The SRE queries reuse the qids of the original test split, so their CLIP text features and MSSE
features must be extracted into **separate** directories and passed explicitly; otherwise the
features of the original queries would be used. For TACoS-SRE:
```bash
python extract_clip_text.py tacos data/tacos/test_SRE.jsonl datasets/tacos/clip_text_features_SRE
python linguistic_knowledge_tacos_v2.py data/tacos/test_SRE.jsonl datasets/semantic_embeddings/tacos-token-level-v2-SRE

bash EventSem/scripts/inference.sh <path/to/model_best.ckpt> data/tacos/test_SRE.jsonl \
    --t_feat_dir datasets/tacos/clip_text_features_SRE \
    --semantic_t_feat_dir datasets/semantic_embeddings/tacos-token-level-v2-SRE
```
Charades-STA-SRE works the same way with `charades` / `linguistic_knowledge_charades_v2.py` and
`data/charades_sta/charades_sta_SRE_test_tvr_format.jsonl`.



## 📈 Model Zoo
We will provide it after the paper is accepted.

## 🙏 Acknowledgments

- Built upon QD-DETR, TR-DETR, and FlashVTG
- Thanks to the creators of TACoS, Charades-STA, and QVHighlights datasets
