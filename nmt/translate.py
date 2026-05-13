import sys
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from nmt.checkpoint import load_training_checkpoint
from nmt.config import get_config, get_weights_path, get_tokenizer_path
from nmt.model import build_transformer
from nmt.dataset import BilingualDataset
from nmt.train import greedy_decode


def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(get_tokenizer_path(config, config["lang_src"])))
    tokenizer_tgt = Tokenizer.from_file(str(get_tokenizer_path(config, config["lang_tgt"])))
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    epoch = str(config["checkpoint_epoch"])
    model_filename = get_weights_path(config, epoch)
    state = load_training_checkpoint(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds = load_dataset(
            config["datasource"],
            f"{config['lang_src']}-{config['lang_tgt']}",
            split="all",
        )
        ds = BilingualDataset(
            ds,
            tokenizer_src,
            tokenizer_tgt,
            config["lang_src"],
            config["lang_tgt"],
            config["seq_len"],
        )
        sentence = ds[idx]["src_text"]
        label = ds[idx]["tgt_text"]
    seq_len = config["seq_len"]

    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        source = source.unsqueeze(0)
        source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).int().to(device)
        output_ids = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
    return tokenizer_tgt.decode(output_ids.tolist())


def main():
    text = translate(sys.argv[1] if len(sys.argv) > 1 else "Ich bin Zubair")
    print(text)


if __name__ == "__main__":
    main()
