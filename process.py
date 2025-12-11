import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import random

test = False  # Enables supervising on each (image, label)
# Change to your working directory
directory = os.path.join("C:", "Users", "ycnew", "OneDrive", "Desktop")
data_path = 'data'
output_path = 'processed_data'
input_size = (256, 256)  # Good option for the task(UNet+detail preservation+light computation)

# augmentation params
aug_number = 5
rot_range = (-30, 30)
scale_range = (0.8, 1.2)
brightness_range = (-0.2, 0.2)
contrast_range = (0.8, 1.2)
std = 10


def augment_pair(img, mask, out_size): # For personal implementations check https://github.com/York-Citynew/Image-Processing-Basics
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    angle = random.uniform(*rot_range)
    scale = random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    img_t  = cv2.warpAffine(img, M, out_size)
    mask_t = cv2.warpAffine(mask, M, out_size)

    # # keep mask binary after warps
    # if mask_t.max() > 1:
    #     mask_t = (mask_t > 127).astype(np.uint8) * 255

    alpha = random.uniform(*contrast_range)
    beta  = random.uniform(brightness_range[0] * 255, brightness_range[1] * 255)
    img_t = cv2.convertScaleAbs(img_t, alpha=alpha, beta=beta)
    noise = np.random.normal(0, std, img_t.shape).astype(np.float32)
    out   = img_t.astype(np.float32) + noise
    img_t = np.clip(out, 0, 255).astype(np.uint8)
    return img_t, mask_t


if not os.path.exists(output_path):
    os.makedirs(output_path)


def process_dataset(src_folder):
    src_path = os.path.join(data_path, src_folder)
    dst_path = os.path.join(output_path, src_folder)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        print('Data is already processed')
        return None
    coco_path = os.path.join(src_path, '_annotations.coco.json')
    coco = COCO(coco_path)
    dst_img_path = os.path.join(dst_path, "images")
    dst_mask_path = os.path.join(dst_path, "masks")
    os.makedirs(dst_img_path)
    os.makedirs(dst_mask_path)
    for _, value in coco.imgs.items():
        img_id = value['id']
        img_file_name = value['file_name']
        src_img_path = os.path.join(src_path, img_file_name)

        # Chose the single-channel grayscale input since RGB channels wouldn't added much information
        img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(
            img, input_size, interpolation=cv2.INTER_NEAREST)

        seg = np.zeros_like(resized_img, dtype=np.uint8)
        ann_ids = coco.getAnnIds(img_id)
        # If an image has multiple annotations combine them(outline preferred picking the last one)
        for ann_id in ann_ids:
            mask = coco.annToMask(coco.anns[ann_id])
            mask_resized = cv2.resize(
                mask, input_size, interpolation=cv2.INTER_NEAREST)
            seg = cv2.add(seg, mask_resized)

        if test:
            while True:
                cv2.imshow('image', resized_img)
                cv2.imshow('segmentation', np.uint8(255*seg))
                if cv2.waitKey() == ord('q'):
                    break
            cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(dst_img_path, img_file_name), resized_img)
        cv2.imwrite(os.path.join(dst_mask_path, img_file_name),
                    np.uint8(255 * seg))

        if src_folder == 'train':
            mask_uint8 = np.uint8((seg > 0) * 255)
            name, ext = os.path.splitext(img_file_name)
            for k in range(aug_number):
                img_aug, mask_aug = augment_pair(resized_img, mask_uint8, input_size)
                cv2.imwrite(os.path.join(dst_img_path, f"{name}_aug{k}{ext}"), img_aug)
                cv2.imwrite(os.path.join(dst_mask_path, f"{name}_aug{k}{ext}"), mask_aug)


# Since test and valid sets are labled wrong, they were manually replaced.
for folder in ['test', 'valid', 'train']:
    print(f"Processing {folder} set")
    process_dataset(folder)


print("Task 1 & 4 are implemented(Except the normalization part because we can't save normalized images)")


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