#公開モジュールのインストール
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
#自作モジュールのインストール
from config.config_loader import load_config
from config.param_set import create_param_yaml
from model.load_transformers import load_model_and_tokenizer, setup_peft, load_and_tokenize_data

#configファイルのパラメータを読み込み
from anyParameters import model, peft, training

# Fine-tuning実行
def main(config_path, dataset_path):
    cfg = load_config(config_path)

    # モデル・Tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = setup_peft(model, cfg)

    # データセット（例：独自データを準備して置き換え）
    dataset = load_and_tokenize_data(cfg, tokenizer, dataset_path)

    # TrainingArguments に YAML の値を渡す
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        fp16=cfg["training"]["fp16"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        push_to_hub=cfg["training"]["push_to_hub"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])


if __name__ == "__main__":
    import os
    import datetime
    
    #configファイルのパラメータを読み込み
    from anyParameters import model, peft, training
    #yamlファイルのパス(自動入力)
    paramdate = datetime.datetime.now().strftime("%Y%m%d")
     
    #yamlファイル作成
    create_param_yaml(paramdate, model, peft, training)
    #yamlファイル参照
    config_folder = os.path.join(os.path.dirname(__file__), "config", paramdate)
    config_files = os.listdir(config_folder)
    config_files = [i for i in config_files if i.endswith(".yaml")]
    #yamlファイルの読み込み
    for config_file in config_files:
        config_path = os.path.join(config_folder, config_file)
        print(f"config_path: {config_path}")
        #Finetuningの実行
        main(config_path)