import argparse
from unsloth import FastLanguageModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from src.prompts import get_train_prompt_v5

MAX_SEQ_LENGTH = 512


def get_model(model_id):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, # 4-bit quantization,
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    # Since your JSON rows contain "<s>" and "</s>" explicitly:
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def apply_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 32,
        lora_dropout = 0, # Unsloth recommends 0 dropout for optimized kernels
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Uses Unsloth's optimized checkpointing
        random_state = 3407,
    )

    found = 0
    print("Freezing vision tower parameters...")
    for name, param in model.named_parameters():
        if "vision" in name.lower():
            found += 1
            param.requires_grad = False

    print(f"Found and froze {found} vision-related parameters.")
    return model

def get_trainer_config():
    sft_config = SFTConfig(
        output_dir = "./results",
        dataset_text_field = "text",
        max_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,

        # --- Training Parameters ---
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        num_train_epochs = 1,
        learning_rate = 1e-4, # 2e-4

        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),

        logging_steps = 25,
        save_strategy = "epoch", #"steps",
        # save_steps = 100,

        warmup_steps = 5,
        optim = "adamw_8bit", # 8-bit optimizer saves even more memory
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,

        # eval_accumulation_steps = 4,
        eval_strategy = "epoch", # "steps",
        # eval_steps = 100,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        args = sft_config,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
    )

    return trainer


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a model.")
    argparser.add_argument("--model", type=str, required=False, default="unsloth/Ministral-3-3B-Instruct-2512", help="The model to train.")
    argparser.add_argument("--dataset", type=str, required=False, default="../../data/processed/train_v9.jsonl", help="The path to the training data file.")
    argparser.add_argument("--version", type=str, required=False, default="v9", help="The version of the training data and model.")
    args = argparser.parse_args()

    print(f"Loading model {args.model}...")
    model, tokenizer = get_model(args.model)

    print("Applying LoRA...")
    model = apply_lora(model)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=3407, shuffle=True) # Use 90% for training, 10% for validation

    dataset = dataset.map(
        lambda x: {
            "text": [get_train_prompt_v5(c, o) for c, o in zip(x["corrupted"], x["original"])]
        }, 
        batched=True
    )

    print("Configuring trainer...")
    trainer = get_trainer_config()

    print("Starting training...")
    trainer.train()

    NEW_MODEL_NAME = f"./models/{args.model.split("/")[1]}-GEC-{args.version}"
    print(f"Saving model to {NEW_MODEL_NAME}...")
    model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)