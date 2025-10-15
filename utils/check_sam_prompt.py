from finetune_sam_slim.cfg import parse_args
from finetune_sam_slim.utils.dataset import Public_dataset



# finetune_sam_slim/utils/check2.py
from pathlib import Path
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from finetune_sam_slim.cfg import parse_args
from finetune_sam_slim.utils.dataset import Public_dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm(img: torch.Tensor):
    # img: (3,H,W) normalized -> (H,W,3) [0,1]
    return ((img * IMAGENET_STD) + IMAGENET_MEAN).clamp(0,1).permute(1,2,0).cpu().numpy()

def viz_points_batches(item: dict, save_path: Path | None = None, show: bool = True):

# check if boxes and points exists in the dataset dict
    has_points = ("points" in item)
    has_boxes  = (item.get("boxes", None) is not None)
    if not (has_points or has_boxes):
        raise RuntimeError("Item has neither 'points' nor 'boxes'.")
    img = item["image"]                  # (3,H,W)
    img_np = denorm(img)
    msk_np = item["mask"].squeeze(0).cpu().numpy()

#point extraction
    if has_points:
        prompts, labels = item["points"]     # (B,N,2), (B,N)
        prompts = prompts.cpu().numpy()
        labels  = labels.cpu().numpy()
        B_pts   = prompts.shape[0]
    else:
        prompts  = labels = None
        B_pts   = 0

# box extraction
    boxes = item.get("boxes", None)
    B_box = 0 if boxes is None else boxes.shape[0]
    if boxes is not None:
        boxes = boxes.cpu().numpy()
    B = max(B_pts, B_box, 1)

    # dynamic grid
    cols = min(6, max(1, math.ceil(math.sqrt(B))))
    rows = math.ceil(B / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.ravel() if isinstance(axes, (list, tuple)) or hasattr(axes, "ravel") else [axes]

    # marker styles per label
    # -1: cross, 0: hollow circle, 1: solid square
    def scatter(ax, x, y, lbl):
        if lbl == -1:
            ax.scatter(x, y, marker="x", s=90, linewidths=2.0)
        elif lbl == 0:
            ax.scatter(x, y, marker="o", s=90, facecolors="none", edgecolors="white", linewidths=2.0)
        else:  # 1
            ax.scatter(x, y, marker="s", s=90, linewidths=1.5)

    for b in range(rows*cols):
        ax = axes[b]
        ax.axis("off")
        if b >= B:  # blank extra cells
            continue

        ax.imshow(img_np, origin="upper")
        masked = np.ma.masked_where(msk_np == 0, msk_np)
        ax.imshow(masked, cmap="viridis", alpha=0.35, interpolation="nearest")
        if has_points and b < B_pts:
            P = prompts[b]  # (N,2)
            L = labels[b]   # (N,)
        else:
            P = L = None
        # draw box if present and valid
        if boxes is not None and b < B_box:
            x0, y0, x1, y1 = boxes[b]
            valid_box = (x1 > x0) and (y1 > y0) and not (x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0)
            if valid_box:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2.0))

        # plot by label so they’re visually distinct
        if has_points and b < B_pts:
            P = prompts[b]
            L = labels[b]
            for lbl in (-1, 0, 1):
                sel = (L == lbl)
                if sel.any():
                    scatter(ax, P[sel, 0], P[sel, 1], lbl)

        # tiny title: counts per label
        if has_points and b < B_pts:
            c_m1 = int((L == -1).sum()); c0 = int((L == 0).sum()); c1 = int((L == 1).sum())
            ax.set_title(f"batch {b}  (+:{c1}  0:{c0}  -1:{c_m1})", fontsize=9)
        else:
            ax.set_title(f"batch {b}", fontsize=9)

    # legend
    legend_elems = [
    Line2D([0],[0], marker='s', linestyle='None', markersize=9, label='pos (1)'),
    Line2D([0],[0], marker='o', linestyle='None', markersize=9, markerfacecolor='none', markeredgecolor='black', label='neg (0)'),
    Line2D([0],[0], marker='x', linestyle='None', markersize=9, label='pad (-1)'),
    Rectangle((0,0), 1, 1, fill=False, label='box'),  # <-- add this
]
    fig.legend(handles=legend_elems, loc='upper right', frameon=True)

    # save/show define path of save according to your preference
    if save_path is None:
        img_name = Path(item.get("img_name", "image")).name
        save_path = Path(f"{Path(img_name).stem}_points_batches.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    print(f"Saved: {save_path}")

def main():
    args = parse_args()
    dataset = Public_dataset(
        args=args,
        img_list=args.img_list,
        phase=args.phase,
        sample_num=args.sample_num,
        crop=args.crop,
        crop_size=args.crop_size,
        targets=args.targets,
        cls=args.cls,
        if_prompt=args.if_prompt,
        neg_prompt_ratio=args.neg_prompt_ratio,
        prompt_type=args.prompt_type,
        label_mapping=args.label_mapping,
        if_spatial=args.if_spatial,
        delete_empty_masks=args.delete_empty_masks,
    )
    item = dataset[1]
    print(item["mask"].shape)
    viz_points_batches(item, save_path=None, show=True)

if __name__ == "__main__":
    main()

