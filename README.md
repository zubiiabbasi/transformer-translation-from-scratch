# Transformer Translation from Scratch

**German → English** neural machine translation on the **[OPUS Books](https://opus.nlpl.eu/Books.php)** parallel corpus, using a **Transformer** encoder–decoder in **PyTorch**. The model is written explicitly (attention, feed-forward blocks, masks, training loop)—not wired through `nn.Transformer`. The reference is [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017).

Code is packaged as **`nmt`** (`pip install -e .`). Artifact paths use **`get_project_root()`** in `nmt/config.py`, so the CLI and notebooks behave consistently on **Windows and Linux** regardless of the process working directory.

---

## Dataset

Parallel text is **[OPUS Books](https://opus.nlpl.eu/Books.php)** (German–English), loaded in `nmt/train.py` with Hugging Face **`datasets`**: `load_dataset("opus_books", "de-en", split="train")`, matching **`datasource`**, **`lang_src`**, and **`lang_tgt`** in **`nmt/config.py`**.

- **Split:** after dropping over-long sentences, the filtered list is split **90% train / 10% validation** (`random_split` in `get_dataset()`).
- **Length filter:** sentence pairs where either side has more than **`seq_len`** word tokens (after whitespace word tokenization) are excluded.
- **Tokenizers:** word-level vocabularies with **`min_frequency=2`**, trained on the full **`train`** split **before** length filtering; saved as **`tokenizer_de.json`** and **`tokenizer_en.json`** at the project root on first training run.

---

## Compared to Vaswani et al. (2017)

### Implemented from the paper

The implementation follows Figure 1 and Section 3 of the paper for the **core Transformer**:

- **Encoder–decoder** stack with **N = 6** layers on each side (**base** width).
- **Multi-head scaled dot-product attention** (encoder self-attention; decoder self-attention with a **causal** mask; **encoder–decoder** cross-attention with queries from the decoder and keys/values from the encoder).
- **Position-wise feed-forward** sublayers (**ReLU**, dropout on sublayer paths).
- **Embeddings** scaled by **√d_model**; **sinusoidal** positional encodings summed into source and target embeddings; **dropout 0.1** on residual paths as in the paper’s setup.
- **Output:** linear projection to the vocabulary and **softmax**; training uses **label smoothing ε = 0.1** (Sec. 5.4).
- **Adam** with **ε = 10^{-9}**; the **learning-rate schedule** and **β₂** differ from the paper (see table).

Default tensor shapes match the **base** model (**d_model = 512**, **h = 8**, **d_ff = 2048**); exact batch length, epochs, and learning rate are set in **`nmt/config.py`**.

### Changed on purpose (defines checkpoint compatibility)

| Topic | This repository | Paper (typical) |
|--------|-----------------|-----------------|
| Residual layout | **Pre-norm** + final stack norm | **Post-norm** (“Add & Norm” in Fig. 1) |
| Optimizer schedule | **Adam**, fixed `lr = 10^{-4}`, default `β₂` | **Noam** schedule, `β₂ = 0.98` |
| Embeddings vs logits | **Separate** target embedding and output linear | **Weight tying** (Sec. 3.4) |
| Data & tokens | **OPUS Books** `de`–`en`, **word-level** tokenizers | WMT’14-style **BPE** benchmark |
| Inference decoding | **Greedy** | **Beam** (e.g. width 4) in reported BLEU tables |

Checkpoints under `weights/` match the **deviation** column. The bullet list above describes what still follows the published architecture and objective, aside from these rows.

---

## Architecture (paper figure)

Figure 1 (encoder left, decoder right, **N**× blocks, linear + softmax). **Norm ordering** in the drawing is post-norm; this code uses **pre-norm** (see table).

![The Transformer: model architecture (Figure 1, Vaswani et al., 2017).](docs/figures/transformer_architecture.png)

---

## Repository contents

| Path | Role |
|------|------|
| `nmt/` | `config`, `model`, `dataset`, `train`, `translate`, `checkpoint` (reload with `map_location`) |
| `train.py`, `translate.py` | Same as `python -m nmt.train` / `python -m nmt.translate` |
| `notebooks/` | Inference, evaluation (`compute_bleu`, SacreBLEU, WER/CER), attention (Altair); first cell sets repo root on `sys.path` and `chdir` |
| `weights/` | `tmodel_*.pt` per training epoch |
| `tokenizer_de.json`, `tokenizer_en.json` | Written on first training run |
| `runs/` | TensorBoard logs |
| `requirements.txt`, `pyproject.toml` | Dependencies; editable install (`requires-python >= 3.10`) |

---

## Installation

**Python 3.10+** recommended (see `pyproject.toml`).

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1` before `pip`, or run  
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m pip install -e .` if activation is blocked.

**Linux / macOS:** `source .venv/bin/activate` then the same `pip` lines.

For Jupyter, select the interpreter from **`.venv`** so notebooks see the same packages.

`tensorboard` is listed in `requirements.txt`. `SummaryWriter` is imported **only inside** `train_model()`, so importing helpers from `nmt.train` does not require TensorBoard until you run training.

---

## How to run

| Goal | Action |
|------|--------|
| Train | `python train.py` |
| TensorBoard | `tensorboard --logdir runs` (from repo root) |
| Translate | `python translate.py "German sentence here."` |
| Inference epoch | `checkpoint_epoch` in `nmt/config.py` |
| Resume training | `preload` in `nmt/config.py` → matching `weights/tmodel_<epoch>.pt` |

---

## Evaluation and results

During training (`nmt/train.py`), a short validation decode is printed each epoch; **CER**, **WER**, and **torchmetrics BLEU** are logged to TensorBoard when logging is on.

Offline metrics: `notebooks/evaluate_model.ipynb` defines **`compute_bleu`** (greedy decode per batch, then corpus-level scores):

```python
compute_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt, config, device, num_batches=100)
```

Raise **`num_batches`** for a more stable estimate (slower run).

**Illustrative** validation numbers on **OPUS Books** (German → English; depend on epoch, seed, hardware):

| Context | BLEU (SacreBLEU) | WER | CER |
|---------|------------------|-----|-----|
| `evaluate_model.ipynb`, `num_batches=100` | **53.65** | **0.6720** | **0.3386** |
| Stronger / longer-trained checkpoint | **~55–56** | **~0.67** | **~0.32** |

Order-of-magnitude **~15 hours** on GPU has been reported for competitive quality; CPU is much slower. These figures are **not** formal benchmark submissions.

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

Dataset download uses bandwidth; training is compute-heavy (**GPU** recommended). Metrics are not guaranteed—use for research and learning.
