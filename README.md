2025年10月8日更新
# 1.ファイル構成
```
AI_FT_Collaborative Project/
├── config/ ：各種パラメーターをまとめたコンフィグファイル(yaml形式)を保存します。
│   ├── ["%Y%m%d"]/ ：param_set.pyで作成されるコンフィグファイルを保存します。フォルダ名は任意の作成年月日を示します。
│   ├── config_loader.py：コンフィグファイルを参照する各種関数が格納されます。
│   └── param_set.py    ：コンフィグファイルを作成する各種関数が格納されます。
│
├── dataset/：データセットを保存します。
│   ├── ORG/：Excel（csv含む）形式のデータセットを保存します。
│   │   ├── QA/     ：QA形式で作成したデータセットを保存します。
│   │   └── Text/   ：原文形式で作成したデータセットを保存時ます。
│   ├── ["%Y%m%d"]/ ：dataset_maker.pyで作成されるデータセットを保存します。フォルダ名は任意の作成年月日を示します。
│   ├── dataset_maker.py：[ORG]からjsonl形式でデータセットを作成する各種関数が格納されます。
│   └── load_dataset.py ：データセットを参照する各種関数が格納されます。
│
├── model/
│   ├── load_transformers.py：モデルとトークナイザーを読み込む各種関数が格納されます。[transformers]モジュールから作成されます（推奨）。
│   └── load_unsloth.py     ：モデルとトークナイザーを読み込む各種関数が格納されます。[unsloth]モジュールから作成されます（現在調整中）。
│
├── utils/  ：その他関数を格納したpythonファイルを保存します（現在なし）。
│
├── anyParameters.py：各種パラメーターを手動で指定します。
├── Finetuning.py   ：ファインチューニングを実施します。
├── README.md       ：このプログラムの使い方を簡潔にまとめたマークダウン形式のファイル。
└── requirements.txt：このプログラムを動作するのに必要な各種モジュールをまとめたテキスト形式のファイル。
```

# 2.プログラムの使い方
1. 必要な各種モジュールのインストール（初回のみ）<br>
    　requirements.txt内に必要なモジュールを整理しています。ターミナルでAI_FT_Collaborative Project(本プログラム)内にアクセスし以下のコマンドを実行して、必要なモジュールをインストールします。<br>
    　なお、２回目以降は実行不要です。
    ```　shell
    # 仮想環境(.venv)をアクティブ化
    pip install -r requirements.txt
    ```
    　なお、仮想環境内で実行することを推奨します。
    ```　shell
    # python3.11環境で仮想環境(.venv)を作成
    conda create -n .venv python=3.11
    
    # 仮想環境(.venv)をアクティブ化
    conda activate .venv
    ```
    <br>
2. 各種パラメーターの設定<br>
    　anyParameters.py内で各種パラメーターを設定することができます。特定のパラメーターで複数の条件がある場合、[]内に複数設定することで、それぞれのコンフィグファイルを作成することができます。
    ```　python
    # 各種パラメーターの設定例
    model = {
    "name":["unsloth/DeepSeek-R1-32B"], # モデル名
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
        "output_dir":["./output"], # モデルの保存先
        "num_train_epochs":[3,4], # エポック数(複数条件)
        "per_device_train_batch_size":[1], # trainバッチサイズ
        "per_device_eval_batch_size":[1], # evalバッチサイズ
        "learning_rate":[2e-4], # 学習率
        "logging_steps":[50], # ロギングのステップ数
        "save_steps":[200], # モデル保存のステップ数（save_strategyが"steps"の場合に有効）
        "evaluation_strategy":["steps"], # 評価の頻度 ("no", "steps", "epoch")
        "fp16":[True], # 16ビット浮動小数点精度で学習するか
        "push_to_hub":[False] # Hugging Face Hubにモデルをプッシュするか
    }
    ```
    <br>
3. Finetuning.pyを実行<br>
    　Finetuning.pyを実行することで、実行日の年月日でコンフィグファイルとデータセットを自動生成し、それらを参照したファインチューニングを開始します。ターミナルで以下のコマンドを実行します。
    ```　shell
    # Finetuning.pyを実行
    python Finetuning.py
    ```