#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, cv2, torch, json
from tqdm import tqdm
from monai.metrics import compute_dice
from segment_anything import sam_model_registry, SamPredictor


data_root   = "data/npy/CT_Abd"
ckpt_path   = "work_dir/SAM/sam_vit_b_01ec64.pth"
out_dir     = "results_sam_zero"
device      = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(out_dir, exist_ok=True)
sam   = sam_model_registry["vit_b"](checkpoint=ckpt_path).to(device)
pred  = SamPredictor(sam)

names = sorted(os.listdir(os.path.join(data_root, "imgs")))
dice_scores = []

for name in tqdm(names, desc="SAM Zero-Shot"):
    img = np.load(os.path.join(data_root, "imgs", name))
    gt  = np.load(os.path.join(data_root, "gts",  name))


    ys, xs = np.where(gt > 0)
    if xs.size == 0:
        mask_pred = np.zeros_like(gt, dtype=np.uint8)
    else:
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        img_bgr = (img * 255).astype(np.uint8)[..., ::-1]
        pred.set_image(img_bgr)
        masks, *_ = pred.predict(box=bbox[None, :], multimask_output=False)
        mask_pred = masks[0].astype(np.uint8)


    np.save(os.path.join(out_dir, name.replace(".npy", "_pred.npy")), mask_pred)


    dice = compute_dice(torch.tensor(mask_pred[None, None]),
                        torch.tensor(gt        [None, None]))
    dice_scores.append(float(dice))

print(f"\nZero-shot Dice (mean ± std over {len(dice_scores)}) = "
      f"{np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")


json.dump({"mean_dice": np.mean(dice_scores),
           "std_dice":  np.std(dice_scores)},
          open(os.path.join(out_dir, "metrics.json"), "w"))
print(f"All masks & metrics saved in: {out_dir}")
