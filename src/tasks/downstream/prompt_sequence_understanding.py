import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable, List

import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    load_from_disk
)
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedModel,
    AutoConfig,
)

# HARD CODED DIRECTORY
from models.prompt_projector import (
    PromptedGenerator,
    DataCollatorForPromptTuning
)

transformers.logging.set_verbosity_info()

METRICS_DIRECTION: Dict[str, str] = {
    "accuracy": "max",
    "f1_score": "max",
    "mcc": "max",
}

def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def dist_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model for sequence understanding")
    
    parser.add_argument("--dataset_name", type=str, required=True, help="Path to the local dataset directory.")
    parser.add_argument("--model_name", type=str, default="GenerTeam/GENERator-eukaryote-1.2b-base", help="HuggingFace model path or name.")
    parser.add_argument("--output_dir", type=str, default="results/sequence_understanding", help="Path to save the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training. Default is higher for prompt tuning.")
    
    # prompt Tuning Arguments
    parser.add_argument("--use_prompt_tuning", action="store_true", help="Enable prompt tuning with a prompt projector.")
    parser.add_argument("--num_prompt_tokens", type=int, default=10, help="Number of prompt tokens to use.")
    parser.add_argument("--num_rna_features", type=int, help="Dimension of the RNA expression vector. Required if --use_prompt_tuning is set.")

    # other training arguments
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problem_type", type=str, default="single_label_classification")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--main_metrics", type=str, default="mcc", choices=["accuracy", "f1_score", "mcc"])
    parser.add_argument("--hf_config_path", type=str, default=None)
    
    args = parser.parse_args()
    
    return args

def setup_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    use_prompt_tuning: bool,
) -> Tuple[DatasetDict, int]:
    """
    Load and prepare dataset for sequence understanding.
    This version robustly finds the sequence column and uses the correct loading function.
    """
    dist_print(f"📚 Loading dataset from local path: {dataset_name}...")
    
    dataset = load_from_disk(dataset_name)
        
    num_labels = 2 

    def _process_function(examples):
        sequences = examples["sequence"]

        # tokenize the sequences
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True, 
            padding=False,
        )
        tokenized["label"] = examples["label"]
        if use_prompt_tuning:
            tokenized["prompt"] = examples["prompt"]
        return tokenized

    dist_print("🧬 Tokenizing DNA sequences and preparing columns...")
    dataset = dataset.map(
        _process_function,
        batched=True,
        num_proc=os.cpu_count() if dataset['train'].num_rows > 1 else 1,
    )

    final_columns = ["input_ids", "attention_mask", "label", "prompt"]
    
    for split in dataset.keys():
        current_columns = dataset[split].column_names
        columns_to_remove = [col for col in current_columns if col not in final_columns]
        if columns_to_remove:
            dist_print(f"🗑️ Removing intermediate columns from '{split}' split: {columns_to_remove}")
            dataset[split] = dataset[split].remove_columns(columns_to_remove)

    if is_main_process() and "train" in dataset:
        print(f"✔️ After processing, final columns are: {dataset['train'].column_names}")

    return dataset, num_labels

def setup_tokenizer(model_name: str, padding_side: str = "right") -> PreTrainedTokenizer:
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_model(model_name: str, num_labels: int, args: argparse.Namespace) -> PreTrainedModel:
    dist_print(f"🤗 Loading model: {model_name} with {num_labels} labels")
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, problem_type=args.problem_type, trust_remote_code=True)
    
    if args.use_prompt_tuning:
        dist_print("🚀 Using Prompt-Tuning model architecture...")
        prompt_tuning_config = {
            "num_rna_features": args.num_rna_features,
            "num_prompt_tokens": args.num_prompt_tokens
        }
        model = PromptedGenerator(
            config=config,
            model_name=model_name,
            prompt_tuning_config=prompt_tuning_config
        )
    else:
        dist_print("Standard fine-tuning model architecture...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dist_print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
    dist_print(f"   Trainable params: {trainable_params / 1e6:.1f}M parameters")
    
    return model

def get_compute_metrics_func(problem_type: str) -> Callable:
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(labels, predictions)
        return {"accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted, "mcc": mcc}
    return _compute_metrics

def setup_training_args(cli_args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cli_args.output_dir,
        per_device_train_batch_size=cli_args.batch_size,
        per_device_eval_batch_size=cli_args.batch_size,
        learning_rate=cli_args.learning_rate,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        seed=cli_args.seed,
        data_seed=cli_args.seed,
        metric_for_best_model=cli_args.main_metrics,
        greater_is_better=METRICS_DIRECTION[cli_args.main_metrics] == "max",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=5000,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=100,
        num_train_epochs=3,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    )

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = setup_tokenizer(args.model_name)
    datasets, num_labels = setup_dataset(args.dataset_name, tokenizer, args.max_length, args.use_prompt_tuning)
    model = setup_model(args.model_name, num_labels, args)
    
    training_args = setup_training_args(args)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    data_collator = DataCollatorForPromptTuning(tokenizer) if args.use_prompt_tuning else DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_func(args.problem_type),
        callbacks=[early_stopping_callback],
    )
    
    dist_print("🏋️ Starting model training...")
    trainer.train()
    dist_print("✅ Training completed.")
    
    test_metrics = trainer.evaluate(datasets["test"])
    
    dist_print("\n" + "=" * 80)
    dist_print(f"🏆 EVALUATION RESULTS FOR {args.model_name} 🏆")
    for metric, value in test_metrics.items():
        if metric.startswith("eval_"):
            dist_print(f"📊 test_{metric[5:].upper()}: {value:.4f}")
    dist_print("=" * 80)
    
    save_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    dist_print(f"💾 Best model saved to {save_model_path}")

if __name__ == "__main__":
    main()
