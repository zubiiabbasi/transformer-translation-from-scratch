from pathlib import Path


def get_project_root() -> Path:
    """Directory containing ``nmt/``, ``weights/``, ``runs/``, etc."""
    return Path(__file__).resolve().parent.parent


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "de",
        "lang_tgt": "en",
        "datasource": "opus_books",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "checkpoint_epoch": "31",
    }


def get_weights_path(config, epoch: str) -> str:
    root = get_project_root()
    name = f"{config['model_basename']}{epoch}.pt"
    return str(root / config["model_folder"] / name)


def get_tokenizer_path(config, language: str) -> Path:
    return get_project_root() / config["tokenizer_file"].format(language)


def get_experiment_dir(config) -> str:
    return str(get_project_root() / config["experiment_name"])
