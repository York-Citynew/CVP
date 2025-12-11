import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

# from transfer_learning import UNet
from model import UNet
# from dmodel import UNet

cv2.setNumThreads(0)  # avoid CPU thread thrash with PyTorch
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_STATS = {
    "train": {"mean": 0.60, "std": 0.18},
    "valid": {"mean": 0.52, "std": 0.15},
    "test":  {"mean": 0.51, "std": 0.13},
}

RESIZE_LOGITS = True

class CrackSegDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.img_dir = f"processed_data/{split}/images"
        self.msk_dir = f"processed_data/{split}/masks"
        all_imgs = sorted(os.listdir(self.img_dir))
        self.files = [f for f in all_imgs if os.path.isfile(
            os.path.join(self.msk_dir, f))]
        stats = DATASET_STATS[split]
        self.mean = np.float32(stats["mean"])
        self.std = np.float32(stats["std"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_p = os.path.join(self.img_dir, fname)
        msk_p = os.path.join(self.msk_dir, fname)
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(
            np.float32) / 255.0
        msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0)

        return img, msk, fname


def make_loaders(batch_size, num_workers):
    train_ds = CrackSegDataset("train")
    val_ds = CrackSegDataset("valid")

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    return train_ld, val_ld


def train(epochs, batch_size, lr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_ld, val_ld = make_loaders(batch_size, 8)
    # If using transfer learning
    # model = UNet(1).to(device)
    # If not
    model = UNet(1, 1, 64).to(device)
    pos_weight = torch.tensor([15.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr)
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for imgs, msks, _ in train_ld:
            imgs, msks = imgs.to(device), msks.to(device)
            logits = model(imgs)
            if RESIZE_LOGITS:
                logits = F.interpolate(logits, size=msks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, msks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
        train_loss = run_loss / len(train_ld.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for imgs, msks, _ in val_ld:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                if RESIZE_LOGITS:
                    logits = F.interpolate(logits, size=msks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, msks)
                val_loss_sum += loss.item() * imgs.size(0)
        val_loss = val_loss_sum / len(val_ld.dataset)
        val_losses.append(val_loss)

        print("Epoch:", epoch, " Train Loss:", round(
            train_loss, 4), " Val Loss:", round(val_loss, 4))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, best_path)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    return best_path


if __name__ == "__main__":
    train(5, 16, 0.002, "checkpoints")