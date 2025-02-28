import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)

# Set logging level for transformers
transformers.logging.set_verbosity_info()


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0) in distributed training.

    Returns:
        bool: True if this is the main process, False otherwise
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for causal language model fine-tuning.

    Returns:
        argparse.Namespace: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for causal language modeling"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Name of the dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="Name of the subset of the dataset (if applicable)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fine_tuning",
        help="Path to save the fine-tuned model",
    )
    # In appropriate situations, we recommend setting --pad_to_multiple_of_six true by default to avoid generating <oov> at the end of sequences.
    parser.add_argument(
        "--pad_to_multiple_of_six",
        action="store_true",
        help="Pad sequences to multiple of 6 with 'A'. ",
    )
    parser.add_argument(
        "--hf_config_path",
        type=str,
        default="configs/hf_configs/fine_tuning.yaml",
        help="Path to the YAML configuration file for HuggingFace Trainer",
    )
    parser.add_argument(
        "--distributed_type",
        type=str,
        default="ddp",
        choices=["ddp", "deepspeed", "fsdp"],
        help="Type of distributed training to use",
    )
    return parser.parse_args()


def setup_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for causal language modeling.

    Args:
        model_name: Name or path of the HuggingFace model

    Returns:
        PreTrainedTokenizer: Configured tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    # Load tokenizer with trust_remote_code to support custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer


def setup_dataset(
    dataset_name: str,
    subset_name: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 512,
    pad_to_multiple_of_six: bool = False,
) -> Dataset:
    """
    Load and prepare dataset for causal language modeling.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        subset_name: Name of the dataset subset (if applicable)
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length for tokenization
        pad_to_multiple_of_six: Whether to pad sequences to multiple of 6

    Returns:
        Dataset: Preprocessed dataset
    """
    dist_print(f"📚 Loading dataset {dataset_name}...")
    start_time = time.time()

    # Load dataset from HuggingFace
    if subset_name is None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)

    dist_print(f"⚡ Dataset loaded in {time.time() - start_time:.2f} seconds")

    if "train" in dataset:
        dataset = dataset["train"]
        dist_print("🔍 Using 'train' split of the dataset")
    elif "test" in dataset:
        dataset = dataset["test"]
        dist_print("🔍 Using 'test' split of the dataset")
    elif "validation" in dataset:
        dataset = dataset["validation"]
        dist_print("🔍 Using 'validation' split of the dataset")
    else:
        raise ValueError("No valid split found in dataset")

    # Process dataset with tokenizer
    def _process_function(examples):
        # Find the correct field containing the sequence
        if "sequence" in examples:
            sequences = examples["sequence"]
        elif "seq" in examples:
            sequences = examples["seq"]
        elif "dna_sequence" in examples:
            sequences = examples["dna_sequence"]
        elif "dna_seq" in examples:
            sequences = examples["dna_seq"]
        elif "text" in examples:
            sequences = examples["text"]
        else:
            raise ValueError(
                "No sequence column found in dataset. Expected 'sequence', 'seq', 'dna_sequence', 'dna_seq', or 'text'."
            )

        # Apply padding to original sequences if requested
        if pad_to_multiple_of_six:
            padded_sequences = []
            for seq in sequences:
                remainder = len(seq) % 6
                if remainder != 0:
                    pad_len = 6 - remainder
                    seq = seq + "A" * pad_len  # Add "A" characters for padding
                padded_sequences.append(seq)
            sequences = padded_sequences

        # Tokenize sequences
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
        )

        return tokenized

    # Apply tokenization to dataset
    dataset = dataset.map(
        _process_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return dataset


def setup_model(model_name: str) -> PreTrainedModel:
    """
    Load and configure model for causal language modeling.

    Args:
        model_name: Name or path of the HuggingFace model

    Returns:
        PreTrainedModel: Configured pre-trained model for causal language modeling
    """
    dist_print(f"🤖 Loading AutoModelForCausalLM from: {model_name}")
    start_time = time.time()

    # Load model with trust_remote_code to support custom models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad_token_id is set
    if model.config.pad_token_id is None and hasattr(model.config, "eos_token_id"):
        model.config.pad_token_id = model.config.eos_token_id

    # Report model size for reference
    total_params = sum(p.numel() for p in model.parameters())
    dist_print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
    dist_print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    return model


def setup_training_args(yaml_path=None, cli_args=None, **kwargs):
    """
    Create a TrainingArguments instance from YAML, CLI arguments, and code arguments.
    Priority: code kwargs > CLI args > YAML config

    Args:
        yaml_path: Path to YAML configuration file
        cli_args: Parsed command line arguments
        **kwargs: Direct arguments that take highest priority

    Returns:
        TrainingArguments: Configured training arguments
    """
    # Start with yaml configuration if provided
    yaml_kwargs = {}
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_kwargs = yaml.safe_load(f)

    # Create a dictionary from CLI arguments
    cli_kwargs = {}
    if cli_args is not None:
        # Add basic training parameters
        if hasattr(cli_args, "output_dir"):
            cli_kwargs["output_dir"] = cli_args.output_dir
        if hasattr(cli_args, "batch_size"):
            cli_kwargs["per_device_train_batch_size"] = cli_args.batch_size
        if hasattr(cli_args, "learning_rate"):
            cli_kwargs["learning_rate"] = cli_args.learning_rate
        if hasattr(cli_args, "gradient_accumulation_steps"):
            cli_kwargs["gradient_accumulation_steps"] = (
                cli_args.gradient_accumulation_steps
            )
        if hasattr(cli_args, "num_train_epochs"):
            cli_kwargs["num_train_epochs"] = cli_args.num_train_epochs

        # Handle distributed training configurations
        if hasattr(cli_args, "distributed_type"):
            if cli_args.distributed_type == "deepspeed":
                cli_kwargs["deepspeed"] = "configs/ds_configs/zero1.json"
            elif cli_args.distributed_type == "fsdp":
                cli_kwargs["fsdp"] = "shard_grad_op auto_wrap"
                cli_kwargs["fsdp_config"] = "configs/ds_configs/fsdp.json"

    # Handle BF16 precision based on GPU capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        cli_kwargs["bf16"] = True

    # Merge all configurations, with priority: kwargs > cli_kwargs > yaml_kwargs
    final_kwargs = {**yaml_kwargs, **cli_kwargs, **kwargs}

    # Add defaults for saving strategy
    if "save_strategy" not in final_kwargs:
        final_kwargs["save_strategy"] = "epoch"

    # Add logging steps if not provided
    if "logging_steps" not in final_kwargs:
        final_kwargs["logging_steps"] = 10

    # Create and return the TrainingArguments instance
    return TrainingArguments(**final_kwargs)


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    args: argparse.Namespace,
) -> Trainer:
    """
    Train the model for causal language modeling.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        dataset: Training dataset
        args: Command line arguments

    Returns:
        Trainer: Trained model trainer
    """
    dist_print("🚀 Setting up training...")
    start_time = time.time()

    # Configure training arguments
    training_args = setup_training_args(yaml_path=args.hf_config_path, cli_args=args)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    dist_print(f"⏱️ Training setup completed in {time.time() - start_time:.2f} seconds")
    dist_print("🏋️ Starting model training...")
    training_start_time = time.time()

    # Train the model
    trainer.train()

    dist_print(
        f"✅ Training completed in {(time.time() - training_start_time) / 60:.2f} minutes"
    )
    return trainer


def save_model(
    trainer: Trainer, tokenizer: PreTrainedTokenizer, output_dir: str
) -> None:
    """
    Save the fine-tuned model and tokenizer.

    Args:
        trainer: Trained model trainer
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the model
    """
    dist_print(f"💾 Saving fine-tuned model to {output_dir}")
    start_time = time.time()

    # Save the model
    trainer.save_model(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    dist_print(f"✅ Model saved in {time.time() - start_time:.2f} seconds")


def display_progress_header() -> None:
    """
    Display a stylized header for the causal language model fine-tuning.
    """
    dist_print("\n" + "=" * 80)
    dist_print("🔥  CAUSAL LANGUAGE MODEL FINE-TUNING PIPELINE  🔥")
    dist_print("=" * 80 + "\n")


def main() -> None:
    """
    Main function to run the causal language model fine-tuning pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tokenizer first
    tokenizer = setup_tokenizer(args.model_name)

    # Load and prepare data
    dataset = setup_dataset(
        args.dataset_name,
        args.subset_name,
        tokenizer,
        args.max_length,
        args.pad_to_multiple_of_six,
    )

    # Initialize model
    model = setup_model(args.model_name)

    # Train model
    trainer = train_model(model, tokenizer, dataset, args)

    # Save fine-tuned model
    save_model(trainer, tokenizer, args.output_dir)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    dist_print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    dist_print("✨ Fine-tuning completed successfully! ✨\n")


if __name__ == "__main__":
    main()
