#configファイルのパラメータ（手動）
model = {
    "name":["cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"], # モデル名
    "max_seq_length":[2048], # 最大シーケンス長
    "dtype":["float16"], # データ型（例：float16, float32, bfloat16）
    "load_in_4bit":[True] # 4bit量子化で読み込むか (True or False)
}

peft = {
    "training_mode":["qlora"], # fine-tuningのモード（"full" or "lora" or "qlora"）
    "r":[8], # LoRAのランク
    "lora_alpha":[16], # LoRAのスケーリングファクター
    "lora_dropout":[0.05], # LoRAのドロップアウト率 
    "target_modules":[
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ] # LoRAを適用するモジュール
}

training = {
    "seed":[42], # 乱数シード
    "output_dir":["./output"], # モデルの保存先
    "num_train_epochs":[3], # エポック数
    "per_device_train_batch_size":[1], # trainバッチサイズ
    "per_device_eval_batch_size":[1], # evalバッチサイズ
    "learning_rate":[2e-4], # 学習率
    "logging_steps":[50], # ロギングのステップ数
    "save_steps":[200], # モデル保存のステップ数（save_strategyが"steps"の場合に有効）
    "evaluation_strategy":["steps"], # 評価の頻度 ("no", "steps", "epoch")
    "fp16":[True], # 16ビット浮動小数点精度で学習するか
    "push_to_hub":[False] # Hugging Face Hubにモデルをプッシュするか
}

dataset = {
    "seed":[42], # 乱数シード
    "train_ratio":[0.8] # 訓練データの割合（例：0.8は80%を訓練、20%を検証に使用）
}