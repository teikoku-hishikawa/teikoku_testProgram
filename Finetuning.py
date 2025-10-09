import os

#公開モジュールのインストール
# from datasets import load_dataset
# from transformers import Trainer, TrainingArguments

#自作モジュールのインストール
from config.config_loader import load_config
from config.param_set import create_param_yaml
from dataset.dataset_maker import DatasetMaker
# from dataset.load_dataset import load_and_tokenize_data
# from model.load_transformers import load_model_and_tokenizer, setup_peft

#configファイルのパラメータを読み込み
from anyParameters import model, peft, training, dataset

# Fine-tuning実行
class Finetuning:
    def __init__(self, paramdate, header_row=1):
        self.paramdate = paramdate
        self.header_row = header_row
        self.config_folder = os.path.join(os.path.dirname(__file__), "config", paramdate)
        self.dataset_folder = os.path.join(os.path.dirname(__file__), "dataset", paramdate)
        
    def main(self):
        #yamlファイル作成
        create_param_yaml(paramdate, model, peft, training, dataset)
        #yamlファイル参照
        config_files = os.listdir(self.config_folder)
        config_files = [i for i in config_files if i.endswith(".yaml")]
        
        for config_file in config_files:
            #yamlファイルの読み込み
            config_path = os.path.join(self.config_folder, config_file)
            cfg = load_config(config_path)
            print(f"config_path: {config_path}")
            
            #データセット作成(既存のデータセットがあればそのまま使用)
            DatasetMakerSet = DatasetMaker(makedate=self.paramdate, header_row=self.header_row, train_ratio=cfg["dataset"]["train_ratio"], seed=cfg["dataset"]["seed"])
            DatasetMakerSet.main()
            dataset_path = DatasetMakerSet.dataset_dir
            
            #Fine-tuning実行(動作確認用にコメントアウト)
            #self.training(cfg, dataset_path)

    def training(self, cfg, dataset_path):
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
    import datetime
    
    #configファイルのパラメータを読み込み
    from anyParameters import model, peft, training
    #yamlファイルのパス(自動入力)
    paramdate = datetime.datetime.now().strftime("%Y%m%d")

    Finetuning(paramdate=paramdate, header_row=1).main()