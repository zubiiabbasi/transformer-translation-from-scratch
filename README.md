# Transformer Translation From Scratch

A minimal, from-scratch implementation of a Transformer model for neural machine translation (NMT) using PyTorch, based on the paper:

> Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS* 30.

## Layout

| Path | Role |
|------|------|
| `nmt/` | Python package: `config`, `model`, `dataset`, `train`, `translate`, checkpoint loading |
| `train.py`, `translate.py` | Thin entry points at the repo root (same as `python -m nmt.train` / `python -m nmt.translate`) |
| `notebooks/` | `inference.ipynb`, `evaluate_model.ipynb`, `attention_visual.ipynb` |
| `weights/` | Saved checkpoints (`tmodel_XX.pt`) |
| `tokenizer_{de,en}.json` | Trained tokenizers at repo root (created on first train) |
| `runs/` | TensorBoard logs |

Paths are resolved from the **repository root** (the parent of `nmt/`), not from the current working directory, so training, the CLI, and notebooks behave the same on Windows and Linux.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`-e .`) lets notebooks and other folders import `nmt` without manual `sys.path` tweaks. The notebooks still add the repo root if you prefer not to install the package.

### Windows notes

- Use PowerShell or Command Prompt from the repo root; paths use `pathlib` so drive letters and backslashes are handled correctly.
- If you trained on Linux with CUDA and load checkpoints on Windows CPU (or vice versa), checkpoints are loaded with `map_location` set to the current device so tensors relocate safely.
- Line endings: Git `core.autocrlf` is optional; Python source is ASCII-only here.

## Usage

**Train**

```bash
python train.py
```

or `python -m nmt.train`.

Checkpoints go to `weights/`. TensorBoard: `tensorboard --logdir runs` (from repo root after `os.chdir` in notebooks, or open a terminal at the repo root).

**Resume training**

In `nmt/config.py`, set `"preload": "09"` (or whatever epoch string matches `tmodel_09.pt`).

**Translate (CLI)**

```bash
python translate.py "Ich bin ein Berliner."
```

Default checkpoint epoch is `nmt/config.py` → `"checkpoint_epoch": "31"`. Change that once instead of editing multiple files.

**Notebooks**

Open files under `notebooks/`. The first cell adds the repo root to `sys.path` and `chdir`s there so `weights/` and `runs/` resolve correctly.

## Configuration

Edit `nmt/config.py` for batch size, languages (`lang_src` / `lang_tgt`), `datasource` (Hugging Face `datasets` name), `checkpoint_epoch`, etc.

If you have a single checkpoint file at the repo root (e.g. `tmodel_31.pt`), move it into `weights/` so it matches `get_weights_path`.

## Model variant (checkpoint compatibility)

The stack matches the **original project / tutorial-style** setup your weights were trained with: **pre-norm** residuals (`x + Dropout(Sublayer(LayerNorm(x)))`), a **final layer norm** on the encoder and decoder stacks, **no weight tying** between target embeddings and the output projection, **fixed Adam learning rate** (`lr` in `nmt/config.py`, default `1e-4`, `eps=1e-9`), and label smoothing `0.1`. That is **not** identical to every detail of Vaswani et al. (2017), but it **does** match saved checkpoints from this repo before any paper-faithfulness experiment.

## Results (example)

After training (German → English), typical validation metrics are in the BLEU ~50–55 range depending on epoch and setup; see the original project notes in git history if needed.
