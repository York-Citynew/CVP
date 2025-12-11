import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from images.transfer_learning import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_STATS = {
    "train": {"mean": 0.60, "std": 0.18},
    "valid": {"mean": 0.52, "std": 0.15},
    "test":  {"mean": 0.51, "std": 0.13},
}
THRESHOLD = 0.5
MODEL_PATH = "checkpoints_tl/best.pt"
OUTPUT_DIR = "eval_min"
BATCH_SIZE = 8
NUM_VISUALS = 6


class CrackSegDataset(Dataset):
    def __init__(self, split="test"):
        self.img_dir = f"processed_data/{split}/images"
        self.msk_dir = f"processed_data/{split}/masks"
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.msk_dir, f))
        ])
        stats = DATASET_STATS[split]
        self.mean = np.float32(stats["mean"])
        self.std = np.float32(stats["std"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname),
                         cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        msk = cv2.imread(os.path.join(self.msk_dir, fname),
                         cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy((msk > 127).astype(
            np.float32)).unsqueeze(0)
        return img, msk, fname


def iou_score(pred_bin, target_bin):
    inter = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()
    return float(inter) / float(union + 0.00001)


@torch.no_grad()
def evaluate(model_path, out_dir, batch_size, num_visuals):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    model = UNet(1).to(DEVICE)
    # model = UNet(1, 1, 64).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    state = state["model"]
    model.load_state_dict(state)
    model.eval()

    ds = CrackSegDataset("test")
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

    scores = []
    for imgs, msks, _ in ld:
        logits = model(imgs.to(DEVICE))
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= THRESHOLD).to(torch.uint8)
        targs = msks.to(torch.uint8)
        for b in range(preds.size(0)):
            p = preds[b, 0].numpy().astype(np.uint8)
            t = targs[b, 0].numpy().astype(np.uint8)
            scores.append(iou_score(p, t))

    mean_iou = float(np.mean(scores))
    print(
        f"Mean IoU on test set: {round(mean_iou, 4)} over {len(scores)} images.")

    for i in range(num_visuals):
        img_t, _, fname = ds[i]
        orig = cv2.imread(os.path.join(ds.img_dir, fname),
                          cv2.IMREAD_GRAYSCALE)
        prob = torch.sigmoid(model(img_t.unsqueeze(
            0).to(DEVICE))).squeeze().cpu().numpy()
        pred = (prob >= THRESHOLD).astype(np.uint8)

        overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        overlay[..., 2] = np.maximum(
            overlay[..., 2], pred * 255)
        cv2.imwrite(os.path.join(
            vis_dir, f"{os.path.splitext(fname)[0]}_overlay.png"), overlay)

    return mean_iou


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH,
        out_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_visuals=NUM_VISUALS,
    )


# Rest of part 3
