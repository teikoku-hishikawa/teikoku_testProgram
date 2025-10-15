import os

#公開モジュールのインストール
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

#自作モジュールのインストール
from config.config_loader import load_config
from config.param_set import create_param_yaml
from dataset.dataset_maker import DatasetMaker
from dataset.load_dataset import load_and_tokenize_data
from model.load_transformers import load_model_and_tokenizer, setup_peft

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
            self.training(cfg, dataset_path)

    def training(self, cfg, dataset_path):
        # モデル・Tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)
        model = setup_peft(model, cfg)

        # データセット（例：独自データを準備して置き換え）
        dataset = load_and_tokenize_data(cfg, tokenizer, dataset_path)

        # データの出力先を確認
        output_dir = os.path.join(cfg["training"]["output_dir"], self.paramdate)
        os.makedirs(output_dir, exist_ok=True)

        # Wandbの記録名
        run_name = self.run_name_set(cfg)

        # TrainingArguments に YAML の値を渡す
        print(TrainingArguments) # 出力結果：<class 'transformers.training_args.TrainingArguments'>
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            logging_steps=cfg["training"]["logging_steps"],
            fp16=cfg["training"]["fp16"],
            #evaluation_strategy=cfg["training"]["evaluation_strategy"],
            push_to_hub=cfg["training"]["push_to_hub"],
            eval_steps=cfg["training"]["eval_steps"],
            save_steps=cfg["training"]["save_steps"],
            save_total_limit=cfg["training"]["save_total_limit"],
            #load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
            metric_for_best_model=cfg["training"]["metric_for_best_model"],
            greater_is_better=cfg["training"]["greater_is_better"],
            report_to=["wandb"],  # ✅ Wandbに記録
            run_name=run_name,  # ✅ 実験名を設定
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(cfg["training"]["output_dir"])
    
    def run_name_set(self, cfg):
        model_name = cfg["model"]["name"]
        if model_name in "DeepSeek":
            if model_name in "14B":
                run_name = f"DeepSeek-R1-14B-Finetune-{self.paramdate}"
            elif model_name in "32B":
                run_name = f"DeepSeek-R1-32B-Finetune-{self.paramdate}"
        else:
            run_name = f"othermodel-Finetune-{self.paramdate}"
        
        return run_name



if __name__ == "__main__":
    import datetime
    import transformers

    print(transformers.__file__)  # 出力結果：/home/teikoku/.conda/envs/.venv/lib/python3.11/site-packages/transformers/__init__.py
    print(transformers.__version__) # 出力結果：4.55.3 ← ver.4.5以上なら"evaluation_strategy"はあるらしい？
    
    #configファイルのパラメータを読み込み
    from anyParameters import model, peft, training
    #yamlファイルのパス(自動入力)
    paramdate = datetime.datetime.now().strftime("%Y%m%d")

    Finetuning(paramdate=paramdate, header_row=1).main()