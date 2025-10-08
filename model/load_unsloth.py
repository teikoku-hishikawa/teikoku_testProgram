import torch

from unsloth import FastLanguageModel #GPTOSSに対応？
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)

# PEFT のバージョンにより関数名が異なる場合があるので安全にインポート
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None  # なければ None にしてフォールバックする

# -------------------
# モデルとトークナイザーの読み込み
# -------------------
def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]

    # dtype 文字列を torch 型に変換（存在しない場合は float16 を使う）
    dtype_str = cfg["model"].get("dtype", "float16")
    dtype = getattr(torch, dtype_str, torch.float16)

    # QLoRA の場合は 4bit でロードする必要があるので強制
    load_in_4bit = bool(cfg["model"].get("load_in_4bit", False))
    if cfg["pref"]["training_mode"].lower() == "qlora":
        load_in_4bit = True

    # bitsandbytes / bnb の追加引数を渡すために、cfg 側で bnb_4bit_kwargs を設定できるようにする
    bnb_kwargs = cfg["model"].get("bnb_4bit_kwargs", {})

    # FastLanguageModel.from_pretrained を使ってモデルと tokenzier を取得
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        **bnb_kwargs,  # 受け付けるなら bitsandbytes の細かいオプション等を渡せる
    )

    # tokenizer の pad_token が未設定なら eos_token を割り当て（Trainerで便利）
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# -------------------
# 学習モードに応じたモデル準備（Full / LoRA / QLoRA 切替）
# -------------------
def setup_peft(model, cfg: dict):
    
    # 学習方針を確認（存在しない場合は qlora を使う）
    mode = cfg["peft"].get("training_mode", "qlora")

    # 学習モードに応じたモデル準備
    # full: 全パラメータを学習可能にして返す
    if mode.lower() == "full":
        # 全パラメータを学習可能にする（注意: 量子化モデルで full fine-tuning は難しい）
        for p in model.parameters():
            p.requires_grad = True
        print("Mode=full: full fine-tuning (all parameters trainable).")
        return model
    # qlora: k-bit 量子化モデルを QLoRA 用に準備
    elif mode.lower() == "qlora":
        print("Mode=qlora: preparing k-bit training (QLoRA flow).")
        if prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)
        else:
            # prepare_model_for_kbit_training が無ければ int8 用関数にフォールバック
            print("Warning: prepare_model_for_kbit_training not available; falling back to prepare_model_for_int8_training.")
            model = prepare_model_for_int8_training(model)
    # lora: int8 量子化モデルを LoRA 用に準備
    elif mode.lower() == "lora":
        print("Mode=lora: preparing int8 training for LoRA.")
        model = prepare_model_for_int8_training(model)
    
    # LoRA 設定を構築（cfg["peft"] から必要な値を取得）
    peft_cfg = LoraConfig(
        r=cfg["peft"]["r"],
        lora_alpha=cfg["peft"]["lora_alpha"],
        lora_dropout=cfg["peft"]["lora_dropout"],
        target_modules=cfg["peft"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    # LoRA 適用
    model = get_peft_model(model, peft_cfg)
    # デバッグ表示（学習対象パラメータの割合など）
    try:
        model.print_trainable_parameters()
    except Exception:
        # .print_trainable_parameters が無い実装もあり得るので安全に
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total} ({trainable/total:.4%})")

    return model