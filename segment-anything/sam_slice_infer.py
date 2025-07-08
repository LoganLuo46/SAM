#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, numpy as np, torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor


DATA_ROOT = "data/npy/CT_Abd"
CKPT      = "work_dir/SAM/sam_vit_b_01ec64.pth"
OUT_DIR   = "results_sam_multi"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

ORGAN_IDS = {
     1: "liver",          2: "right_kidney",  3: "spleen",       4: "pancreas",
     5: "aorta",          6: "IVC",           7: "right_adrenal",8: "left_adrenal",
     9: "gallbladder",   10: "esophagus",    11: "stomach",     12: "duodenum",
    13: "left_kidney",
}


os.makedirs(OUT_DIR, exist_ok=True)
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(DEVICE)
predictor = SamPredictor(sam)

dice_bank = {oid: [] for oid in ORGAN_IDS}
eps = 1e-6

img_files = sorted(os.listdir(os.path.join(DATA_ROOT, "imgs")))


for fname in tqdm(img_files, desc="SAM inference"):
    img = np.load(os.path.join(DATA_ROOT, "imgs", fname))  
    gt  = np.load(os.path.join(DATA_ROOT, "gts",  fname)) 
    img_bgr = (img * 255).astype(np.uint8)[..., ::-1]
    predictor.set_image(img_bgr)

    for oid, organ_name in ORGAN_IDS.items():
        mask_gt = (gt == oid).astype(np.uint8)
        if mask_gt.sum() == 0:          # 此 slice 没有该器官
            continue

        ys, xs = np.where(mask_gt)
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

        masks, *_ = predictor.predict(box=bbox[None, :], multimask_output=False)
        mask_pred = masks[0].astype(np.uint8)

        inter = (mask_pred & mask_gt).sum()
        union = mask_pred.sum() + mask_gt.sum()
        dice  = (2 * inter + eps) / (union + eps)
        dice_bank[oid].append(dice)


per_mean = {ORGAN_IDS[oid]: (np.mean(v) if v else None) for oid, v in dice_bank.items()}
macro_mean = np.mean([v for v in per_mean.values() if v is not None])

print("\n===== Zero-Shot SAM Dice per Organ =====")
print("{:<3s}{:<15s}{}".format("#", "Organ", "Dice"))
print("-" * 30)
for idx, (name, val) in enumerate(per_mean.items(), 1):
    print(f"{idx:<3d}{name:<15s}{val:.4f}" if val is not None else f"{idx:<3d}{name:<15s}N/A")
print("-" * 30)
print(f"{'Macro-average':<18s}{macro_mean:.4f}")


json_path = os.path.join(OUT_DIR, "multi_metrics.json")
txt_path  = os.path.join(OUT_DIR, "dice_table.txt")

json.dump({"per_organ": per_mean, "macro": macro_mean}, open(json_path, "w"))

with open(txt_path, "w") as f:
    f.write("===== Zero-Shot SAM Dice per Organ =====\n")
    f.write("{:<3s}{:<15s}{}\n".format("#", "Organ", "Dice"))
    f.write("-" * 30 + "\n")
    for idx, (name, val) in enumerate(per_mean.items(), 1):
        f.write(f"{idx:<3d}{name:<15s}{val if val is not None else 'N/A'}\n")
    f.write("-" * 30 + "\n")
    f.write(f"Macro-average       {macro_mean:.4f}\n")

print(f"\nJSON saved to {json_path}\nTable saved to {txt_path}")
