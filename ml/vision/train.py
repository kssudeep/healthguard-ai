"""
ml/vision/train.py

Fine-tunes DenseNet-121 on NIH ChestX-ray14.
Logs all experiments to MLflow.

Usage:
    python ml/vision/train.py --epochs 30 --batch_size 32 --lr 0.001
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from agents.vision_agent.agent import ChestXRayModel, PATHOLOGY_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """NIH ChestX-ray14 dataset loader."""

    def __init__(self, csv_path: str, images_dir: str, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.images_dir) / row["Image Index"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(
            [row.get(cls, 0) for cls in PATHOLOGY_CLASSES], dtype=torch.float32
        )
        return img, labels


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("healthguard-vision-training")

    with mlflow.start_run(run_name=f"densenet121-lr{args.lr}-ep{args.epochs}"):
        mlflow.log_params(vars(args))

        # Model
        model = ChestXRayModel(num_classes=14, pretrained=True).to(device)

        # Freeze early layers (transfer learning)
        for name, param in model.model.named_parameters():
            if "denseblock4" not in name and "classifier" not in name:
                param.requires_grad = False

        # Data
        train_ds = ChestXRayDataset(args.train_csv, args.images_dir, TRAIN_TRANSFORM)
        val_ds = ChestXRayDataset(args.val_csv, args.images_dir, VAL_TRANSFORM)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Loss + optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_auc = 0.0

        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs)
                    val_loss += criterion(preds, labels).item()
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Per-class AUC
            aucs = []
            for i, cls in enumerate(PATHOLOGY_CLASSES):
                if all_labels[:, i].sum() > 0:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    aucs.append(auc)
                    mlflow.log_metric(f"auc_{cls}", auc, step=epoch)

            mean_auc = np.mean(aucs)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "mean_auc": mean_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Mean AUC: {mean_auc:.4f}"
            )

            # Save best model
            if mean_auc > best_auc:
                best_auc = mean_auc
                os.makedirs("ml/vision/weights", exist_ok=True)
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "best_auc": best_auc},
                    "ml/vision/weights/densenet121_chestxray14.pth",
                )
                mlflow.log_artifact("ml/vision/weights/densenet121_chestxray14.pth")
                logger.info(f"  ✅ New best AUC: {best_auc:.4f}")

        mlflow.log_metric("best_mean_auc", best_auc)
        logger.info(f"Training complete. Best mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_csv", default="data/train_labels.csv")
    parser.add_argument("--val_csv", default="data/val_labels.csv")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--mlflow_uri", default="http://localhost:5000")
    args = parser.parse_args()
    train(args)
