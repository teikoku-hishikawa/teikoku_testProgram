from datasets import load_dataset

# -----------------------------
# データセットの読み込み
# -----------------------------
def load_and_tokenize_data(cfg, tokenizer, dataset_path):
    
    dataset_param = {
        "name": "jsonl",
        "data_files":{
            "train": f"{dataset_path}/train.jsonl",
            "validation": f"{dataset_path}/valid.jsonl",
        },
        "columns":{
            "input": "input",
            "output": "output",
        },
    }
    
    ds = load_dataset(
        dataset_param["name"],
        dataset_param["data_files"]
    )

    # Alpacaフォーマット: input + output
    inp_col = dataset_param["columns"]["input"]
    out_col = dataset_param["columns"]["output"]

    def tokenize_fn(example):
        prompt = example[inp_col]
        full_text = prompt + example[out_col]
        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_seq_length"],
        )

        # モデルが output 部分のみを学習するようにラベルを設定
        prompt_len = len(tokenizer(prompt).input_ids)
        labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
        labels = labels[: cfg["model"]["max_seq_length"]]  # 長さを揃える

        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = ds.map(tokenize_fn, batched=False)
    return tokenized_ds