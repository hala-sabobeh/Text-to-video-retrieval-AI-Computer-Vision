# Text-to-Video Retrieval — AI & Computer Vision

A deep learning system that retrieves relevant videos from a corpus given a natural language text query. Built with PyTorch, it uses a dual-encoder architecture combining a sentence transformer for text and X-CLIP for video, trained end-to-end with contrastive loss.



---

## How It Works

The model encodes text queries and video frames into a shared embedding space. At retrieval time, a query is encoded and compared against precomputed video embeddings using cosine similarity, returning the most semantically relevant videos ranked by score.

- **Text encoder:** `sentence-transformers/all-MiniLM-L6-v2`
- **Video encoder:** `microsoft/xclip-base-patch32`
- **Loss:** Symmetric contrastive loss (CLIP-style)
- **Optimizer:** AdamW with cosine annealing scheduler

---

## Project Structure

```
.
├── models/
│   └── text_video_model.py     # Dual-encoder model definition
├── utils/
│   ├── data_loader.py          # Dataset and DataLoader
│   └── evaluation.py           # Metrics and search utilities
├── config.py                   # All hyperparameters and paths
├── train.py                    # Training loop
├── retrieval.py                # Evaluation and interactive search
├── video_preview_retrieval.py  # Video preview interface
├── simple_test.py              # Quick sanity check
├── requirements.txt
└── report.pdf
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/hala-sabobeh/Text-to-video-retrieval-AI-Computer-Vision.git
cd Text-to-video-retrieval-AI-Computer-Vision
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Prepare your data**

Place your files under `data/` following this structure:
```
data/
├── annotations.txt       # Text–video pairs
├── video_corpus.csv      # Video metadata
└── videos/               # Raw video files
```

---

## Usage

### Training

```bash
python train.py
```

Trains for 10 epochs with batch size 16. Checkpoints are saved to `checkpoints/`, and the best model (by R@1) is saved as `checkpoints/best_model.pth`. A training loss plot is also generated.

### Evaluation

```bash
python retrieval.py --mode evaluate --checkpoint checkpoints/best_model.pth
```

Prints a full metrics report including Recall@1/5/10, Median Rank, Mean Rank, mAP, and retrieval speed.

### Interactive Search

```bash
python retrieval.py --mode interactive --checkpoint checkpoints/best_model.pth
```

Launches a query loop — type any text description and get the top 10 most relevant video IDs with similarity scores. Type `quit` to exit.

### Single Query

```bash
python retrieval.py --checkpoint checkpoints/best_model.pth --query "a person riding a bicycle"
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| R@1 / R@5 / R@10 | % of queries where the correct video appears in the top K results |
| Median Rank (MedR) | Median rank of the correct video across all queries |
| Mean Rank (MeanR) | Average rank of the correct video across all queries |
| mAP | Mean Average Precision across all queries |
| Retrieval Time | Time per query in milliseconds |

---

## Configuration

All settings are in `config.py`:

```python
TEXT_ENCODER   = 'sentence-transformers/all-MiniLM-L6-v2'
VIDEO_ENCODER  = 'microsoft/xclip-base-patch32'
EMBEDDING_DIM  = 128
BATCH_SIZE     = 16
NUM_EPOCHS     = 10
LEARNING_RATE  = 1e-4
NUM_FRAMES     = 1
MAX_SAMPLES    = 1000   # subset training
```

> **Windows users:** `NUM_WORKERS` is set to `0` by default to avoid multiprocessing issues.

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
timm>=0.9.0
einops>=0.6.0
```
