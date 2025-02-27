import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable

import numpy as np
import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
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
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
)

METRICS_DIRECTION: Dict[str, str] = {
    "accuracy": "max",
    "f1_score": "max",
    "mcc": "max",
    "f1_max": "max",
    "auprc_micro": "max",
    "mse": "min",
    "mae": "min",
    "r2": "max",
    "pearson": "max",
}


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

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for sequence understanding fine-tuning.

    Returns:
        argparse.Namespace: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for sequence understanding"
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
        help="Batch size per GPU for training and evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model",
    )
    parser.add_argument(
        "--padding_and_truncation_side",
        type=str,
        default="right",
        choices=["right", "left"],
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sequence_understanding",
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=10,
        help="Number of folds for cross-validation (if splitting locally)",
    )
    parser.add_argument(
        "--fold_id",
        type=int,
        default=0,
        help="Fold ID for cross-validation (if splitting locally)",
    )
    parser.add_argument(
        "--main_metrics",
        type=str,
        default="mcc",
        help="Main metric for early stopping and model selection",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluations with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="single_label_classification",
        choices=[
            "single_label_classification",
            "multi_label_classification",
            "regression",
        ],
        help="Problem type for the task",
    )
    parser.add_argument(
        "--hf_config_path",
        type=str,
        default="configs/hf_configs/downstream.yaml",
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


def load_and_prepare_data(
    dataset_name: str,
    subset_name: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 16384,
    problem_type: str = "single_label_classification",
    seed: int = 42,
    num_folds: int = 0,
    fold_id: int = -1,
) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], int]:
    """
    Load and prepare dataset for sequence understanding.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        subset_name: Name of the dataset subset (if applicable)
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length for tokenization
        problem_type: Type of problem (classification or regression)
        seed: Random seed for reproducibility
        num_folds: Number of folds for cross-validation (0 to use existing splits)
        fold_id: Current fold ID when using cross-validation

    Returns:
        Tuple of (preprocessed dataset, number of labels)
    """
    dist_print(f"📚 Loading dataset {dataset_name}...")
    start_time = time.time()

    # Load dataset from HuggingFace
    if subset_name is None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)
    dist_print(f"⚡ Dataset loaded in {time.time() - start_time:.2f} seconds")

    if problem_type == "single_label_classification":
        max_label = max(dataset["train"]["label"])
        num_labels = max_label + 1
    elif problem_type == "multi_label_classification":
        assert isinstance(dataset["train"]["label"][0], list)
        assert isinstance(dataset["train"]["label"][0][0], float)
        num_labels = len(dataset["train"]["label"][0])
    elif problem_type == "regression":
        if isinstance(dataset["train"]["label"][0], list):
            num_labels = len(dataset["train"]["label"][0])
        elif isinstance(dataset["train"]["label"][0], float):
            num_labels = 1
        else:
            raise NotImplementedError(
                "Regression with non-float labels is not supported yet."
            )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    assert num_labels is not None, "Number of labels could not be determined."

    if not any(x in dataset for x in ["validation", "valid", "val"]):
        # No validation set, split train into train and validation
        assert (
            num_folds > 0 and fold_id >= 0
        ), "No validation set found. Please provide a valid setting for cross-validation."

        dist_print(
            f"Performing {num_folds}-fold cross-validation (using fold {fold_id})"
        )

        kfold = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        train_data_list = list(dataset["train"])
        splits = list(kfold.split(train_data_list))
        train_idx, valid_idx = splits[fold_id]
        dataset["validation"] = dataset["train"].select(valid_idx)
        dataset["train"] = dataset["train"].select(train_idx)

    # Process dataset
    def _process_function(examples):
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
        )
        tokenized["attention_mask"] = [
            [1] * len(input_id) for input_id in tokenized["input_ids"]
        ]
        tokenized["label"] = examples["label"]
        return tokenized

    dataset = dataset.map(
        _process_function,
        batched=True,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ["input_ids", "label"]
        ],
    )

    return dataset, num_labels


def setup_tokenizer(
    model_name: str, padding_and_truncation_side: str
) -> PreTrainedTokenizer:
    """
    Load tokenizer for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        padding_and_truncation_side: Side for padding and truncation (left or right)

    Returns:
        PreTrainedTokenizer: Tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = padding_and_truncation_side
    tokenizer.truncation_side = padding_and_truncation_side
    tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer


def setup_model(model_name: str, problem_type: str, num_labels: int) -> PreTrainedModel:
    """
    Load model for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        problem_type: Type of problem (classification or regression)
        num_labels: Number of labels for the task

    Returns:
        PreTrainedModel: Pre-trained model for sequence classification
    """
    dist_print(
        f"🤗 Loading AutoModelForSequenceClassification from: {model_name} with {num_labels} labels"
    )
    start_time = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        trust_remote_code=True,
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Check model size
    total_params = sum(p.numel() for p in model.parameters())
    dist_print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
    dist_print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    return model


def get_compute_metrics_func(problem_type: str, num_labels: int) -> Callable:
    """
    Get the appropriate compute_metrics function based on problem type.

    Args:
        problem_type: Type of problem (classification or regression)
        num_labels: Number of labels for the task

    Returns:
        Callable: A function to compute metrics for the given problem type
    """

    def _compute_metrics_single_label_classification(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = (predictions == labels).mean()
        f1 = f1_score(labels, predictions, average="weighted")
        mcc = matthews_corrcoef(labels, predictions)

        return {"accuracy": accuracy, "f1_score": f1, "mcc": mcc}

    def _compute_metrics_multi_label_classification(eval_pred):
        predictions, labels = eval_pred

        return {
            "f1_max": f1_max(torch.tensor(predictions), torch.tensor(labels)),
            "auprc_micro": area_under_prc(
                torch.tensor(predictions).flatten(),
                torch.tensor(labels).long().flatten(),
            ),
        }

    def _compute_metrics_regression(eval_pred):
        logits, labels = eval_pred
        predictions = logits.squeeze()
        labels = labels.squeeze()

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, num_labels)
        if labels.ndim == 1:
            labels = labels.reshape(-1, num_labels)

        results = {}

        if num_labels > 1:
            label_names = [f"label_{i}" for i in range(num_labels)]

            for idx, label in enumerate(label_names):
                pred = predictions[:, idx]
                true = labels[:, idx]

                # MSE
                mse = np.mean((pred - true) ** 2)
                results[f"mse_{label}"] = mse

                # MAE
                mae = np.mean(np.abs(pred - true))
                results[f"mae_{label}"] = mae

                # R²
                y_mean = np.mean(true)
                ss_tot = np.sum((true - y_mean) ** 2)
                ss_res = np.sum((true - pred) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
                results[f"r2_{label}"] = r2

                # Pearson
                x_mean = np.mean(pred)
                numerator = np.sum((pred - x_mean) * (true - y_mean))
                denominator = np.sqrt(
                    np.sum((pred - x_mean) ** 2) * np.sum((true - y_mean) ** 2)
                )
                pearson = numerator / denominator if denominator != 0 else float("nan")
                results[f"pearson_{label}"] = pearson

        total_mse = np.mean((predictions - labels) ** 2)
        total_mae = np.mean(np.abs(predictions - labels))
        total_y_mean = np.mean(labels)
        total_ss_tot = np.sum((labels - total_y_mean) ** 2)
        total_ss_res = np.sum((labels - predictions) ** 2)
        total_r2 = (
            1 - (total_ss_res / total_ss_tot) if total_ss_tot != 0 else float("nan")
        )
        total_pred_mean = np.mean(predictions)
        total_numerator = np.sum(
            (predictions - total_pred_mean) * (labels - total_y_mean)
        )
        total_denominator = np.sqrt(
            np.sum((predictions - total_pred_mean) ** 2)
            * np.sum((labels - total_y_mean) ** 2)
        )
        total_pearson = (
            total_numerator / total_denominator
            if total_denominator != 0
            else float("nan")
        )

        results["mse"] = total_mse
        results["mae"] = total_mae
        results["r2"] = total_r2
        results["pearson"] = total_pearson

        return results

    def area_under_prc(pred, target):
        """
        Area under precision-recall curve (PRC).

        Parameters:
            pred (Tensor): predictions of shape :math:`(n,)`
            target (Tensor): binary targets of shape :math:`(n,)`
        """
        order = pred.argsort(descending=True)
        target = target[order]
        precision = target.cumsum(0) / torch.arange(
            1, len(target) + 1, device=target.device
        )
        auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
        return auprc

    def f1_max(pred, target):
        """
        F1 score with the optimal threshold.

        This function first enumerates all possible thresholds for deciding positive and negative
        samples, and then pick the threshold with the maximal F1 score.

        Parameters:
            pred (Tensor): predictions of shape :math:`(B, N)`
            target (Tensor): binary targets of shape :math:`(B, N)`
        """
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order)
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
        return all_f1.max()

    if problem_type == "single_label_classification":
        return _compute_metrics_single_label_classification
    elif problem_type == "multi_label_classification":
        return _compute_metrics_multi_label_classification
    elif problem_type == "regression":
        return _compute_metrics_regression
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    args: argparse.Namespace,
) -> Trainer:
    """
    Train the model for sequence understanding.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        datasets: Dictionary containing train, validation, and test datasets
        args: Command line arguments

    Returns:
        Trainer: Trained model trainer
    """
    dist_print("🚀 Setting up training...")
    start_time = time.time()

    # Load default training arguments
    parser = HfArgumentParser([TrainingArguments])
    training_args = parser.parse_yaml_file(args.hf_config_path, True)[0]
    training_args.output_dir = args.output_dir
    training_args.per_device_train_batch_size = args.batch_size
    training_args.per_device_eval_batch_size = args.batch_size
    training_args.learning_rate = args.learning_rate
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.greater_is_better = METRICS_DIRECTION[args.main_metrics] == "max"
    training_args.seed = training_args.data_seed = args.seed
    if "mode" in training_args.lr_scheduler_kwargs:
        training_args.lr_scheduler_kwargs["mode"] = METRICS_DIRECTION[args.main_metrics]
    if args.distributed_type == "deepspeed":
        training_args.deepspeed = "configs/ds_configs/zero1.json"
    if args.distributed_type == "fsdp":
        training_args.fsdp = "shard_grad_op auto_wrap"
        training_args.fsdp_config = "configs/ds_configs/fsdp.json"
    if torch.cuda.get_device_capability()[0] >= 8:
        training_args.bf16 = True
    else:
        # Some models will be overflow when using fp16 (using fp32)
        pass

    # Setup early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=get_compute_metrics_func(
            args.problem_type, model.config.num_labels
        ),
        callbacks=[early_stopping_callback],
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


def evaluate_model(trainer: Trainer, test_dataset: Dataset) -> Dict[str, float]:
    """
    Evaluate the fine-tuned model on the test dataset.

    Args:
        trainer: Trained model trainer
        test_dataset: Test dataset

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    dist_print("📊 Evaluating model on test dataset...")
    start_time = time.time()

    # Run evaluation
    test_results = trainer.evaluate(test_dataset)

    dist_print(f"⏱️ Evaluation completed in {time.time() - start_time:.2f} seconds")

    # Prepare readable results
    metrics = {k.replace("eval_", "test_"): v for k, v in test_results.items()}

    return metrics


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
    Display a stylized header for the sequence understanding fine-tuning.
    """
    dist_print("\n" + "=" * 80)
    dist_print("🔥  SEQUENCE UNDERSTANDING FINE-TUNING PIPELINE  🔥")
    dist_print("=" * 80 + "\n")


def main() -> None:
    """
    Main function to run the sequence fine-tuning pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tokenizer first
    tokenizer = setup_tokenizer(args.model_name, args.padding_and_truncation_side)

    # Load and prepare data
    datasets, num_labels = load_and_prepare_data(
        args.dataset_name,
        args.subset_name,
        tokenizer,
        args.max_length,
        args.problem_type,
        args.seed,
        args.num_folds,
        args.fold_id,
    )

    # Now initialize model with correct number of labels
    model = setup_model(args.model_name, args.problem_type, num_labels)

    # Train model
    trainer = train_model(model, tokenizer, datasets, args)

    # Evaluate on test set
    test_metrics = evaluate_model(trainer, datasets["test"])

    # Print results
    dist_print("\n" + "=" * 80)
    dist_print(f"🏆 EVALUATION RESULTS FOR {args.model_name} 🏆")
    dist_print("=" * 80)
    for metric, value in test_metrics.items():
        if metric.startswith("test_"):
            dist_print(f"📊 {metric[5:].upper()}: {value:.4f}")
    dist_print("=" * 80)

    # Save fine-tuned model
    save_model(trainer, tokenizer, os.path.join(args.output_dir, "best_model"))

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    dist_print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    dist_print("✨ Fine-tuning completed successfully! ✨\n")


if __name__ == "__main__":
    main()
