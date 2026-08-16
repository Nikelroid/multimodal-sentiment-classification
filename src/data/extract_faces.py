"""Precompute face crops for MSCTD frames with MTCNN (largest face per frame).

Missing output file for an index means no face was detected; downstream code
falls back to a blank crop.

Usage:
  python src/data/extract_faces.py --images_dir <frames> --out_dir <crops>
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pathlib import Path

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_prob", type=float, default=0.90)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    from facenet_pytorch import MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(select_largest=True, post_process=False, device=device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(Path(args.images_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    images = [p for p in images if int(p.stem) % args.num_shards == args.shard]
    found = skipped = 0
    for n, path in enumerate(images):
        dst = out / path.name
        if dst.exists():
            found += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and probs[0] is not None and probs[0] >= args.min_prob:
                x1, y1, x2, y2 = boxes[0]
                # pad the box 20% so the crop keeps forehead/chin context
                w, h = x2 - x1, y2 - y1
                x1, y1 = max(0, x1 - 0.2 * w), max(0, y1 - 0.2 * h)
                x2, y2 = min(img.width, x2 + 0.2 * w), min(img.height, y2 + 0.2 * h)
                crop = img.crop((x1, y1, x2, y2)).resize((args.size, args.size))
                crop.save(dst, quality=90)
                found += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
        if (n + 1) % 2000 == 0:
            print(f"{n + 1}/{len(images)} processed | faces: {found} | no-face: {skipped}", flush=True)
    print(f"DONE {args.images_dir}: {found} faces / {len(images)} frames ({skipped} without)")


if __name__ == "__main__":
    main()
