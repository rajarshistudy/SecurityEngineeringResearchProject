import os
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen2.5-1.5B-Instruct")
DATA_DIR = os.getenv("DATA_DIR", "data/cronus_sft_hf")
OUT_DIR = os.getenv("OUT_DIR", "cronus-sft")
SEED = int(os.getenv("SEED", "42"))

def main():
    set_seed(SEED)
    ds = load_from_disk(DATA_DIR)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_8bit=True
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()