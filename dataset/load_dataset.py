from datasets import load_dataset

# -----------------------------
# データセットの読み込み
# -----------------------------
def load_and_tokenize_data(cfg, tokenizer, dataset_path):
    
    dataset = {

    }
    
    ds = load_dataset(
        cfg["dataset"]["name"]
    )

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_seq_length"],
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    return tokenized_ds