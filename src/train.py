"""
Training script for sentiment classifier.

Usage:
    python src/train.py --config configs/train_config.yaml
    python src/train.py --config configs/train_config.yaml --epochs 5 --lr 0.001
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_loader import load_imdb_data, split_data
from src.dataset import SentimentDataset, create_data_loaders
from src.model import SentimentClassifier, create_model_from_config
from src.preprocessing import TextPreprocessor
from src.utils import AverageMeter, EarlyStopping, get_device, load_config, save_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--experiment-name", type=str, default="sentiment-classifier")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=1.0):
    """Train for one epoch."""
    model.train()
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("accuracy")
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        loss_meter.update(loss.item(), input_ids.size(0))
        acc_meter.update(acc.item(), input_ids.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    loss_meter = AverageMeter("loss")
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            loss_meter.update(loss.item(), input_ids.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_seed(config["training"]["random_seed"])
    device = get_device(config["device"]["use_cuda"], config["device"]["cuda_device"])
    logger.info(f"Using device: {device}")

    # Setup MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or f"train_{timestamp}"):
        # Log parameters
        mlflow.log_params({
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "embedding_dim": config["model"]["embedding_dim"],
            "hidden_dim": config["model"]["hidden_dim"],
        })

        # Load data
        logger.info("Loading data...")
        train_texts, train_labels = load_imdb_data(config["data"]["train_path"])
        train_texts, train_labels, val_texts, val_labels = split_data(
            train_texts, train_labels,
            validation_split=config["training"]["validation_split"],
            random_seed=config["training"]["random_seed"],
        )
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

        # Build preprocessor
        preprocessor = TextPreprocessor(
            max_vocab_size=config["data"]["max_vocab_size"],
            max_seq_length=config["data"]["max_seq_length"],
            min_word_freq=config["data"]["min_word_freq"],
        )
        preprocessor.fit(train_texts)
        vocab_size = preprocessor.get_vocab_size()

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_texts, train_labels, val_texts, val_labels,
            preprocessor, batch_size=config["training"]["batch_size"],
        )

        # Initialize model
        model = create_model_from_config(config, vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"], mode="min")
        writer = SummaryWriter(log_dir=str(Path(config["paths"]["log_dir"]) / timestamp))

        best_val_loss = float("inf")
        save_path = Path(config["paths"]["model_save_dir"]) / "sentiment_model"

        # Training loop
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(1, config["training"]["epochs"] + 1):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device,
                                        config["training"]["gradient_clip"])
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])

            logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                       f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

            # Log to TensorBoard and MLflow
            for name, value in {**{f"train_{k}": v for k, v in train_metrics.items()},
                               **{f"val_{k}": v for k, v in val_metrics.items()}}.items():
                writer.add_scalar(name, value, epoch)
            mlflow.log_metrics({**{f"train_{k}": v for k, v in train_metrics.items()},
                               **{f"val_{k}": v for k, v in val_metrics.items()}}, step=epoch)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                model.save_pretrained(str(save_path))
                preprocessor.save(str(save_path))

            if early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_texts, test_labels = load_imdb_data(config["data"]["test_path"])
        test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

        best_model = SentimentClassifier.from_pretrained(str(save_path), device)
        test_metrics = evaluate(best_model, test_loader, criterion, device)

        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({"training_time_minutes": (time.time() - start_time) / 60})
        mlflow.log_artifacts(str(save_path), artifact_path="model")

        writer.close()
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
