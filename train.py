import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_all_sentences(dataset, language):
    sentences = []
    for example in dataset:
        if language in example:
            sentences.append(example[language])
    return sentences

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config, language):
    dataset_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    #keep 90% of the data for training and 10% for validation
    train_dataset_size = int(len(dataset_raw) * 0.9)
    validation_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw , validation_dataset_raw= random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

    