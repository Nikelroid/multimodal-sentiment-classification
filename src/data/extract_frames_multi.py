"""Extract frames at 25/50/75% of each MELD clip listed in a manifest.

A single mid-frame is a noisy view of an utterance (the labeled speaker may
be off-screen for that instant); three spread frames let the face branch
train on more views and vote at inference.

Writes <processed>/frames3/<split>/<key>_p{25,50,75}.jpg. Run
extract_faces.py on that directory afterwards for the MTCNN crops.

Usage:
  python src/data/extract_frames_multi.py --extracted .../extracted \
      --processed .../processed --split train --workers 16
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.data.preprocess_meld import FFMPEG, SPLITS, clip_duration

POSITIONS = (25, 50, 75)


def process(task):
    mp4, out_dir = Path(task[0]), Path(task[1])
    targets = [out_dir / f"{mp4.stem}_p{p}.jpg" for p in POSITIONS]
    if all(t.exists() for t in targets):
        return 1
    dur = clip_duration(mp4)
    if dur is None or dur <= 0:
        return 0
    for p, dst in zip(POSITIONS, targets):
        if not dst.exists():
            subprocess.run([FFMPEG, "-y", "-v", "error",
                            "-ss", f"{dur * p / 100:.2f}", "-i", str(mp4),
                            "-frames:v", "1", "-q:v", "3", str(dst)],
                           capture_output=True)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    _, video_dir = SPLITS[args.split]
    df = pd.read_csv(Path(args.processed) / f"manifest_{args.split}.csv")
    out_dir = Path(args.processed) / "frames3" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(Path(args.extracted) / video_dir / f"{k}.mp4"), str(out_dir))
             for k in df.key]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        done = sum(pool.map(process, tasks, chunksize=16))
    print(f"{args.split}: frames for {done}/{len(tasks)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
