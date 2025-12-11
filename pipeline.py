import os, cv2, torch, numpy as np
from dmodel import UNet
# from model import UNet

MODEL = "checkpoints_dropout/best.pt"
SIZE = 256
MEAN, STD = 0.51, 0.13
THR = 0.5

def predict_mask(img_path, out_path="mask.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(1, 1, 64).to(device).eval()
    state = torch.load(MODEL, map_location=device)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=True)

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    x = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x)[None, None].to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= THR).astype(np.uint8) * 255

    cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    IMAGE_PATH = "p.jpg"
    OUTPUT_PATH = "mask.png"
    predict_mask(IMAGE_PATH, OUTPUT_PATH)
