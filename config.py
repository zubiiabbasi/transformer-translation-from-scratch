from pathlib import Path

def get_config():
    return {
        "batch_size" : 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len" : 350, 
        "d_model" : 512,
        "lang_src": "de",  # Source language
        "lang_tgt": "en",  # Target language
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None, # Path to pretrained weights, if any e.g: "09"
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_path(config, epoch: str):
    model_folder =  config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".")/model_folder/model_filename)