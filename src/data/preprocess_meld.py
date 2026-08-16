"""Preprocess MELD.Raw into per-utterance training artifacts.

For every utterance clip (diaD_uttU.mp4) this extracts:
  - 16 kHz mono wav          -> <out>/audio/<split>/diaD_uttU.wav
  - a frame at 40% of clip   -> <out>/frames/<split>/diaD_uttU.jpg
and writes <out>/manifest_<split>.csv joining them with the MELD labels,
sentiment mapped to the project encoding (neutral: 0, negative: 1,
positive: 2) with the 7-class emotion kept alongside.

Clips that ffmpeg cannot decode (a handful are corrupt in the official
release) are logged and dropped from the manifest.

Usage:
  python src/data/preprocess_meld.py --raw .../MELD.Raw \
      --extracted .../extracted --out .../processed --split train --workers 16
"""
import argparse
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def ffmpeg_exe():
    """System ffmpeg when available, else imageio-ffmpeg's bundled static
    binary (compute nodes here can't reliably load an ffmpeg module)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = ffmpeg_exe()

SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev": ("dev_sent_emo.csv", "dev_splits_complete"),
    "test": ("test_sent_emo.csv", "output_repeated_splits_test"),
}
SENTIMENT_LABEL = {"neutral": 0, "negative": 1, "positive": 2}


def clip_duration(mp4):
    """Duration in seconds, parsed from ffmpeg's stderr banner."""
    p = subprocess.run([FFMPEG, "-i", str(mp4)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", p.stderr)
    if not m:
        return None
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def process_clip(task):
    mp4, wav, jpg = (Path(p) for p in task)
    if wav.exists() and jpg.exists():
        return (mp4.stem, "ok")
    dur = clip_duration(mp4)
    if dur is None or dur <= 0:
        return (mp4.stem, "undecodable")
    ok = True
    if not wav.exists():
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-i", str(mp4),
             "-ac", "1", "-ar", "16000", "-vn", str(wav)],
            capture_output=True)
        ok &= r.returncode == 0 and wav.exists()
    if not jpg.exists():
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-ss", f"{0.4 * dur:.2f}",
             "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(jpg)],
            capture_output=True)
        ok &= r.returncode == 0 and jpg.exists()
    return (mp4.stem, "ok" if ok else "ffmpeg_failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="MELD.Raw dir (holds the label CSVs)")
    ap.add_argument("--extracted", required=True, help="dir the three tars were extracted into")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    csv_name, video_dir = SPLITS[args.split]
    try:
        df = pd.read_csv(Path(args.raw) / csv_name)
    except UnicodeDecodeError:
        df = pd.read_csv(Path(args.raw) / csv_name, encoding="latin-1")

    out = Path(args.out)
    wav_dir, jpg_dir = out / "audio" / args.split, out / "frames" / args.split
    wav_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)

    rows, tasks, missing = [], [], 0
    for _, r in df.iterrows():
        key = f"dia{int(r.Dialogue_ID)}_utt{int(r.Utterance_ID)}"
        mp4 = Path(args.extracted) / video_dir / f"{key}.mp4"
        if not mp4.exists():
            missing += 1
            continue
        rows.append({"key": key, "label": SENTIMENT_LABEL[r.Sentiment.strip().lower()],
                     "sentiment": r.Sentiment.strip().lower(),
                     "emotion": r.Emotion.strip().lower(), "text": r.Utterance})
        tasks.append((str(mp4), str(wav_dir / f"{key}.wav"), str(jpg_dir / f"{key}.jpg")))

    print(f"{args.split}: {len(tasks)} clips ({missing} listed in CSV but missing on disk)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        status = dict(pool.map(process_clip, tasks, chunksize=16))

    bad = sorted(k for k, s in status.items() if s != "ok")
    if bad:
        print(f"dropped {len(bad)} undecodable clips: {', '.join(bad[:20])}"
              + (" ..." if len(bad) > 20 else ""))
    manifest = pd.DataFrame([r for r in rows if status.get(r["key"]) == "ok"])
    manifest = manifest.drop_duplicates(subset="key")
    dest = out / f"manifest_{args.split}.csv"
    manifest.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(manifest)} utterances | "
          f"label counts: {manifest['sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
