"""
Training script for sentiment classifier.

Usage:
    python src/train.py --config configs/train_config.yaml
    python src/train.py --config configs/train_config.yaml --verbose
    python src/train.py --config configs/train_config.yaml --epochs 20 --lr 0.0005
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    get_data_statistics,
    load_imdb_data,
    print_data_statistics,
    split_data,
)
from src.dataset import SentimentDataset, create_data_loaders
from src.model import SentimentClassifier, create_model_from_config
from src.preprocessing import TextPreprocessor
from src.utils import (
    AverageMeter,
    EarlyStopping,
    ensure_dir,
    get_device,
    load_config,
    save_config,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sentiment classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    log_every: int = 100,
) -> dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        gradient_clip: Gradient clipping value
        log_every: Log every N steps
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("accuracy")
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
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
        
        progress_bar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.4f}",
        })
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter("loss")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
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
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    config = load_config(args.config)
    
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        config["training"]["random_seed"] = args.seed
    
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config["paths"]["log_dir"]) / f"train_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=str(log_file))
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    seed = config["training"]["random_seed"]
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    device = get_device(
        use_cuda=config["device"]["use_cuda"],
        cuda_device=config["device"]["cuda_device"],
    )
    logger.info(f"Using device: {device}")
    
    logger.info("Loading training data...")
    train_texts, train_labels = load_imdb_data(config["data"]["train_path"])
    
    stats = get_data_statistics(train_texts, train_labels)
    print_data_statistics(stats)
    
    train_texts, train_labels, val_texts, val_labels = split_data(
        train_texts,
        train_labels,
        validation_split=config["training"]["validation_split"],
        random_seed=seed,
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    logger.info("Building vocabulary...")
    preprocessor = TextPreprocessor(
        max_vocab_size=config["data"]["max_vocab_size"],
        max_seq_length=config["data"]["max_seq_length"],
        min_word_freq=config["data"]["min_word_freq"],
    )
    preprocessor.fit(train_texts)
    
    vocab_size = preprocessor.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    train_loader, val_loader = create_data_loaders(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        preprocessor,
        batch_size=config["training"]["batch_size"],
    )
    
    logger.info("Initializing model...")
    model = create_model_from_config(config, vocab_size)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )
    
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        mode="min",
    )
    
    writer = SummaryWriter(log_dir=str(Path(config["paths"]["log_dir"]) / timestamp))
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    logger.info("Starting training loop...")
    start_time = time.time()
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
        logger.info("-" * 40)
        
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config["training"]["gradient_clip"],
            log_every=config["logging"]["log_every_n_steps"],
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics["loss"])
        
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )
        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, "
            f"ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )
        
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/train", train_metrics["f1"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("ROC-AUC/val", val_metrics["roc_auc"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            
            save_path = Path(config["paths"]["model_save_dir"]) / "sentiment_model"
            model.save_pretrained(str(save_path))
            preprocessor.save(str(save_path))
            
            logger.info(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
        
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total training time: {total_time / 60:.2f} minutes")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    save_path = Path(config["paths"]["model_save_dir"]) / "sentiment_model"
    save_config(config, str(save_path / "training_config.yaml"))
    
    logger.info("\nLoading test data for final evaluation...")
    test_texts, test_labels = load_imdb_data(config["data"]["test_path"])
    
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    
    best_model = SentimentClassifier.from_pretrained(str(save_path), device)
    test_metrics = evaluate(best_model, test_loader, criterion, device)
    
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Results")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info("=" * 60)
    
    writer.close()
    
    logger.info(f"\nModel saved to: {save_path}")
    logger.info(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
