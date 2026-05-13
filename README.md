# Transformer-Based Neural Machine Translation from Scratch

A compact research-style codebase that implements an encoder–decoder **Transformer** (Vaswani et al., 2017) for **German–English** translation using **PyTorch**, without high-level sequence APIs beyond standard modules. The implementation is intended for **reproducible experimentation**, **checkpoint portability** (including across Windows and Linux), and **pedagogical clarity**.

---

## Abstract

We present a self-contained neural machine translation (NMT) system built around a multi-head attention Transformer trained on parallel sentences from the **OPUS Books** corpus (`de`–`en`). Source and target sides use **word-level** tokenizers (Hugging Face `tokenizers`). Training optimizes a **masked cross-entropy** objective with **label smoothing**; validation reports **BLEU**, **word error rate (WER)**, and **character error rate (CER)** via `torchmetrics`, with optional **SacreBLEU** in the evaluation notebook. Inference supports **greedy left-to-right decoding**. Repository paths are anchored to the **project root** so scripts, the command-line interface, and Jupyter notebooks agree on locations for weights, tokenizers, and TensorBoard logs regardless of the shell working directory.

---

## 1. Introduction

The dominant architecture for sequence-to-sequence modeling in machine translation is the **Transformer**, which replaces recurrence with **self-attention** and **position-wise feed-forward** layers, enabling efficient parallel training over long contexts. This repository instantiates that paradigm for a **bilingual** setting: encoder states summarize the source sentence; the decoder **attends** to those states while modeling the target distribution autoregressively.

The implementation follows the spirit of Vaswani et al. (2017) while adopting a **pre-norm residual** formulation and **fixed learning-rate Adam** training that match **existing checkpoints** produced from this project (see Section 4 for the precise architectural contract).

**Reference.**

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). In *Advances in Neural Information Processing Systems* (NeurIPS), 30.

---

## 2. Architecture

### 2.1 Transformer model (from Vaswani et al., 2017)

The canonical encoder–decoder structure is illustrated in the figure below (reproduced from the original publication). The **encoder** (left) maps an input sequence of symbol representations to a sequence of continuous representations; the **decoder** (right) generates an output sequence of symbols one element at a time. Each stack consists of **N** repeated layers of multi-head attention and position-wise feed-forward blocks, with residual connections and layer normalization as drawn.

![The Transformer: model architecture (Figure 1 in Vaswani et al., 2017).](docs/figures/transformer_architecture.png)

**Figure 1.** The Transformer—model architecture. *Source: Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.* Reproduced here for exposition; refer to the [arXiv preprint](https://arxiv.org/abs/1706.03762) for the authoritative version.

### 2.2 Default hyperparameters in this repository

The implementation follows the figure at a high level: **N = 6** layers, **h = 8** attention heads, **d_model = 512**, **d_ff = 2048**, dropout **0.1**, sinusoidal positional encodings, and masked multi-head attention in the decoder. Training uses **label smoothing** (0.1) on the target vocabulary.

### 2.3 Residual ordering (implementation vs figure)

The paper’s diagram uses **post-norm** residuals (“Add & Norm” after each sublayer). This codebase uses **pre-norm** residuals (`x + Dropout(Sublayer(LayerNorm(x)))`) plus a **final layer norm** on each stack, as documented in Section 4, so that saved checkpoints remain valid. Data flow is otherwise aligned with the figure: encoder output supplies keys and values to the decoder’s middle attention sublayer; the top **Linear** and **Softmax** correspond to the output projection and next-token distribution in `nmt/model.py`.

### 2.4 Pipeline (textual)

**Corpus and tokenization:** OPUS Books (`de`–`en`) via Hugging Face `datasets`; word-level tokenizers written to `tokenizer_de.json` and `tokenizer_en.json`. **Training:** `BilingualDataset` batches through the encoder–decoder, cross-entropy with label smoothing, Adam updates; checkpoints under `weights/`, TensorBoard under `runs/`. Paths are resolved from the repository root (`nmt/config.py`). **Inference:** greedy autoregressive decoding from a trained checkpoint.

---

## 3. Repository layout

| Path | Description |
|------|-------------|
| `nmt/` | Installable package: configuration, `model`, `dataset`, `train`, `translate`, checkpoint I/O |
| `train.py`, `translate.py` | Root entry points; equivalent to `python -m nmt.train` and `python -m nmt.translate` |
| `notebooks/` | `inference.ipynb`, `evaluate_model.ipynb`, `attention_visual.ipynb` |
| `weights/` | Serialized training state (`tmodel_XX.pt`) |
| `tokenizer_{de,en}.json` | Fitted word-level tokenizers (created on first training run) |
| `runs/` | TensorBoard event files |
| `pyproject.toml` | Editable install metadata (`pip install -e .`) |
| `requirements.txt` | Runtime dependencies (PyTorch, `datasets`, `tokenizers`, metrics, TensorBoard, etc.) |

---

## 4. Architectural and training contract (checkpoint compatibility)

The following choices define the **computational graph** assumed by weights produced from this repository:

- **Residuals:** pre-norm, `x + Dropout(Sublayer(LayerNorm(x)))` within each sublayer; **final layer norm** after the full encoder and decoder stacks.
- **Embeddings and logits:** **no weight tying** between target embeddings and the output projection.
- **Optimization:** **Adam** with fixed learning rate `lr` (default `10^{-4}`), `eps = 10^{-9}`, and **label smoothing** `0.1` on the target vocabulary.
- **Positional encoding:** sinusoidal, non-learned; same scheme on source and target.

Hyperparameters (`batch_size`, `seq_len`, `num_epochs`, language pair, `datasource`) are centralized in `nmt/config.py`. The default data source is **`opus_books`** for the configured language pair.

---

## 5. Installation and environment

From the repository root:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On Windows, if script activation fails, install with the venv interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

**TensorBoard** is listed in `requirements.txt` and is imported **only inside** `train_model()` so that importing `get_model`, `get_dataset`, or `greedy_decode` in notebooks does not require TensorBoard unless training is executed.

**Cross-platform paths:** artifact paths are built from the repository root (`get_project_root()` in `nmt/config.py`), which stabilizes behavior under Windows and Linux regardless of the process current working directory.

---

## 6. Usage (reproducibility)

**Training**

```bash
python train.py
```

**Resuming** from an epoch checkpoint: set `"preload": "<epoch>"` in `nmt/config.py` (e.g. `"09"` for `weights/tmodel_09.pt`).

**TensorBoard**

```bash
tensorboard --logdir runs
```

(run from the repository root, or after the notebooks’ `os.chdir` to the root).

**CLI translation**

```bash
python translate.py "Ich bin ein Berliner."
```

Default inference epoch is `checkpoint_epoch` in `nmt/config.py` (e.g. `"31"`).

**Notebooks** under `notebooks/` bootstrap `sys.path` and `chdir` to the project root so relative paths to `weights/` and `runs/` remain valid when the kernel’s working directory is not the repository root.

---

## 7. Evaluation protocol

### 7.1 Validation during training

`nmt/train.py` prints a small number of **greedy** source / target / predicted strings each epoch. When TensorBoard logging is enabled, **CER**, **WER**, and **torchmetrics BLEU** are written for that validation slice.

### 7.2 Offline metrics (`evaluate_model.ipynb`)

The notebook `notebooks/evaluate_model.ipynb` loads the same checkpoint and validation dataloader as inference, then aggregates **SacreBLEU** corpus BLEU together with **torchmetrics** WER and CER over many batches. The helper **`compute_bleu`** runs greedy decoding per example, builds reference / hypothesis lists, and prints corpus-level scores.

Typical end of the notebook (after defining `compute_bleu`):

```python
compute_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt, config, device, num_batches=100)
```

- **`num_batches`** caps how many validation batches are scored (default `100` in the notebook); increase for a more stable estimate at the cost of runtime.
- **Outputs:** printed **BLEU** (SacreBLEU), **WER**, and **CER**; illustrative runs on this project have landed near **BLEU ~53–56**, **WER ~0.67**, **CER ~0.33** on German–English OPUS-style validation, depending on epoch and checkpoint.

### 7.3 Attention visualization

`notebooks/attention_visual.ipynb` renders **Altair** heatmaps of encoder self-attention, decoder self-attention, and encoder–decoder cross-attention for selected layers and heads (values reflect the last forward pass that filled each module’s `attention_scores`).

---

## 8. Empirical note

Numbers in Section 7.2 are **illustrative** of strong checkpoints on this codebase, not a fixed benchmark. Hardware, `num_batches`, and data subsampling all shift reported metrics.

---

## 9. Citation

If you use this repository or build on its code, please cite the original Transformer paper and, if appropriate, acknowledge this codebase:

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

## License and disclaimer

Training consumes network bandwidth (dataset download) and compute (GPU recommended). Model quality and metric estimates are **not warranted**; use at your own discretion for research and education.
