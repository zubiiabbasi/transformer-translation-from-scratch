# Transformer Translation from Scratch

End-to-end **German → English** neural machine translation with a **Transformer** encoder–decoder built in **PyTorch** (no canned `nn.Transformer` stack—attention, FFN, and training loop are explicit). Data come from **OPUS Books** (`de`–`en`) via Hugging Face `datasets`; tokenization is **word-level** (`tokenizers`). The repo is organized as an installable **`nmt`** package, with paths fixed to the **project root** so training, the CLI, and notebooks behave the same on **Windows and Linux** even if the working directory changes.

---

## What this repository contains

| Area | Details |
|------|---------|
| **Model** | Multi-head self-/cross-attention, position-wise ReLU FFN, sinusoidal positional encodings, dropout. Default stack: **N = 6**, **h = 8**, **d_model = 512**, **d_ff = 2048**, dropout **0.1**. |
| **Training** | Masked cross-entropy with **label smoothing 0.1**, **Adam** (`lr = 1e-4`, `eps = 1e-9`), TensorBoard scalars, per-epoch checkpoints in `weights/` (`tmodel_XX.pt`). |
| **Inference** | **Greedy** left-to-right decoding; CLI `translate.py` and `nmt/translate.py`. |
| **Notebooks** | `inference.ipynb` (load model, validate, translate), `evaluate_model.ipynb` (SacreBLEU + WER/CER over many batches), `attention_visual.ipynb` (Altair attention heatmaps). |
| **Config** | Hyperparameters, language pair, `datasource`, `checkpoint_epoch`, `preload` for resume—all in `nmt/config.py`. |

**Implementation note (important for checkpoints):** Residuals use **pre-norm** (`x + Dropout(Sublayer(LayerNorm(x)))`) and a **final layer norm** on each stack; there is **no weight tying** between target embeddings and the output linear. That matches **weights trained with this codebase**; it is not identical to every detail of Vaswani et al. (2017), where the published diagram uses post-norm “Add & Norm”.

---

## Architecture (paper figure)

The following figure is the standard Transformer diagram from the original paper (encoder left, decoder right, **N** repeated blocks, multi-head attention + FFN, then linear + softmax for logits).

![The Transformer: model architecture (Figure 1, Vaswani et al., 2017).](docs/figures/transformer_architecture.png)

*Figure 1 — The Transformer. Source: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.*

---

## Project layout

| Path | Role |
|------|------|
| `nmt/` | Package: `config`, `model`, `dataset`, `train`, `translate`, `checkpoint` helpers |
| `train.py`, `translate.py` | Short wrappers; same as `python -m nmt.train` / `python -m nmt.translate` |
| `notebooks/` | Inference, evaluation, attention plots |
| `weights/` | Checkpoints `tmodel_XX.pt` |
| `tokenizer_de.json`, `tokenizer_en.json` | Built on first training run |
| `runs/` | TensorBoard logs |
| `requirements.txt` | Dependencies (torch, datasets, tokenizers, torchmetrics, tensorboard, sacrebleu, pandas, altair, …) |
| `pyproject.toml` | Enables `pip install -e .` |

---

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If activation is blocked, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m pip install -e .`.

`SummaryWriter` is imported **only when training** starts, so you can import `get_model` / `get_dataset` in notebooks without TensorBoard unless you run the full training loop.

---

## How to run

| Goal | Command / action |
|------|------------------|
| Train | `python train.py` |
| Resume | Set `"preload": "09"` (etc.) in `nmt/config.py` to match `weights/tmodel_09.pt` |
| TensorBoard | `tensorboard --logdir runs` (from repo root) |
| Translate | `python translate.py "Your German sentence."` |
| Default checkpoint for inference / notebooks | `checkpoint_epoch` in `nmt/config.py` (e.g. `"31"`) |

Open notebooks under `notebooks/`; the first code cell adds the repo root to `sys.path` and `chdir`s there so `weights/` and `runs/` resolve correctly.

---

## Evaluation and results

**During training** (`nmt/train.py`): a few validation examples are printed each epoch; **CER**, **WER**, and **torchmetrics BLEU** are logged to TensorBoard when logging is enabled.

**Offline evaluation** (`notebooks/evaluate_model.ipynb`): after loading the model and `val_dataloader`, the notebook defines `compute_bleu`, which greedy-decodes each batch, accumulates references and hypotheses, and prints **SacreBLEU** corpus BLEU plus **WER** and **CER** from torchmetrics. Typical call:

```python
compute_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt, config, device, num_batches=100)
```

`num_batches` limits runtime (raise it for a tighter estimate).

**Example numbers** obtained with this project (German → English, OPUS-style validation; exact values depend on epoch, seed, and hardware):

| Setting | BLEU (SacreBLEU) | WER | CER |
|---------|------------------|-----|-----|
| `evaluate_model.ipynb`, `num_batches=100` (example run) | **53.65** | **0.6720** | **0.3386** |
| Stronger checkpoint / longer training (order of magnitude) | **~55–56** | **~0.67** | **~0.32** |

These are **illustrative**, not a formal benchmark submission. The original training log for this codebase mentioned on the order of **~15 hours** to reach competitive validation quality on GPU; CPU training is much slower.

---

## Citation

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```

---

## Disclaimer

Dataset download uses network bandwidth; training is compute-heavy (GPU recommended). Metrics and model quality are **not guaranteed**; use for research and learning.
