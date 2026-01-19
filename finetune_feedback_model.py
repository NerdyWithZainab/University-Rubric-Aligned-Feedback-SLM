"""
finetune_feedback_model.py
Complete fine-tuning pipeline using Unsloth
"""

import torch
from torch.cuda import is_available
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    model_name = "unsloth/Phi-3-mini-4k-instruct"
    max_seq_length = 2048
    load_in_4bit = True

    # LoRA  settings
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0

    # Training settings
    learning_rate = 2e-4
    num_epochs = 3
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 4
    warmup_steps = 10

    # Paths
    data_dir = './data_processed'
    output_dir = './feedback_model_phi3'

    #Logging
    logging_steps = 10
    eval_steps = 50
    save_steps = 50
    save_total_limit = 3

config = Config()

# ============================================================================
# DATA LOADING AND FORMATTING
# ============================================================================

def load_data():
    """Load the processed datasets"""
    train_dataset = load_dataset('json',data_files=f'{config.data_dir}/train.json',split='train')
    val_dataset = load_dataset('json', data_files=f'{config.data_dir}/val.json', split='train')

    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    return train_dataset, val_dataset

def formatting_prompts_func(examples, tokenizer):
    """
    Format examples using chat template
    This converts our instruction/response format into the model's expected format
    """

    instructions = examples["instruction"]
    responses = examples["response"]
    texts = []

    for instruction, response in zip(instructions, responses):
        # Create chat messages
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model():
    """Load and configure the base model with LoRA"""

    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    print(f"✓ Loaded {config.model_name}")   

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    print(f"✓ LoRA configured (r={config.lora_r}, alpha={config.lora_alpha})")

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        )

    print("✓ Chat template applied")     

    return model, tokenizer

# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training function"""

    # Load model
    model, tokenizer = setup_model()

    # Load data
    train_dataset, val_dataset = load_data()

    print("\n" + "="*80)
    print("FORMATTING DATASETS")
    print("FORMATTING DATASETS")
    print("="*80)

    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    print("✓ Datasets formatted")

    # Show GPU stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024 , 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 /1024, 3)
        print(f"\n✓ GPU: {gpu_stats.name}")
        print(f"✓ Max memory: {max_memory} GB")
        print(f"✓ Reserved memory: {start_gpu_memory} GB")

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate= config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb"
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )

    # Train the model
    print("\nStarting training...")
    trainer_stats = trainer.train()

    # Show stats
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 /1024 / 1024, 3)
        print(f"✓ Peak GPU memory: {used_memory} GB")

    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    # Save LoRA adapters
    model.save_pretrained_merged(
        f"{config.output_dir}/merged_model",
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"✓ Merged model saved to {config.output_dir}/merged_model")

    return model , tokenizer

if __name__ == '__main__':
    model, tokenizer = train()
    print("\n🎉 Fine-tuning complete!")