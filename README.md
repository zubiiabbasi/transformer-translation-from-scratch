# Transformer Translation From Scratch


A minimal, from-scratch implementation of a Transformer model for neural machine translation (NMT) using PyTorch, based on the seminal paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*, 30.

This project demonstrates how to build, train, and evaluate a sequence-to-sequence translation model from scratch.

## Features
- Custom Transformer encoder-decoder architecture
- Tokenization using HuggingFace Tokenizers
- Training/validation on OPUS Books
- BLEU, Word Error Rate, and Character Error Rate evaluation
- TensorBoard logging

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model:**
   ```bash
   python train.py
   ```
   - Model checkpoints are saved in `weights/`.
   - Training/validation metrics are logged to TensorBoard.

2. **Monitor training:**
   ```bash
   tensorboard --logdir runs/
   ```

## Configuration
Edit `config.py` to change hyperparameters, language pairs, or experiment names.

## File Overview
- `train.py` — Training loop
- `model.py` — Transformer model
- `dataset.py` — Data utilities
- `config.py` — Config/hyperparameters
- `translate.py` — Translate sentences
- `requirements.txt` — Dependencies

## Notes
- Change `lang_src`/`lang_tgt` in `config.py` according to your desire langugae


