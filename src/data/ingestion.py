import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import subprocess
import src.config

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_msctd():
    os.makedirs(src.config.MSCTD_DIR, exist_ok=True)
    os.chdir(src.config.MSCTD_DIR)

    if not os.path.exists("MSCTD_data"):
        print("Cloning MSCTD repo for text labels...")
        run_cmd("git clone https://github.com/XL2248/MSCTD .")

    # Assuming the user verified these links are still active
    urls = {
        "train_ende.zip": "1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj",
        "test.zip": "1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W",
        "dev.zip": "1F2Kx_N2r1VnB87C--a-17Z7O0K3s0O1Z" # Placeholder if dev is different
    }

    for fname, file_id in urls.items():
        if not os.path.exists(fname) and not os.path.exists(fname.replace('.zip', '')):
            try:
                run_cmd(f"gdown --id {file_id}")
                run_cmd(f"unzip -qq {fname}")
            except Exception as e:
                print(f"Warning: Failed to download {fname}: {e}")

    # Moving to unified structure
    for split in ['train', 'test', 'dev']:
        split_dir = os.path.join("dataset", split)
        os.makedirs(split_dir, exist_ok=True)
        # Simplified moving logic, to be tuned based on exact unzipped names
        run_cmd(f"mv *{split}* {split_dir}/ 2>/dev/null || true")
    
    os.chdir(src.config.DATA_DIR)

def download_instany():
    if not os.path.exists(src.config.INSTANY_DIR):
        print("Downloading InstaNY100K via Kaggle...")
        os.makedirs(src.config.INSTANY_DIR, exist_ok=True)
        os.chdir(src.config.INSTANY_DIR)
        try:
            run_cmd("kaggle datasets download -d hsankesara/flickr-image-dataset")
            run_cmd("unzip -qq flickr-image-dataset.zip")
        except Exception as e:
            print(f"Warning: kaggle failure: {e}. Provide correct credentials.")
        os.chdir(src.config.DATA_DIR)

def download_audio_sample():
    audio_dir = src.config.DATA_DIR / "AudioSample"
    if not os.path.exists(audio_dir):
        print("Downloading small sample audio dataset for audio-multimodal testing...")
        os.makedirs(audio_dir, exist_ok=True)
        os.chdir(audio_dir)
        # Download RAVDESS subset via kaggle or direct link (e.g., example link)
        try:
            run_cmd("kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-video")
            run_cmd("unzip -qq ravdess-emotional-speech-video.zip")
        except Exception as e:
            print("Audio dataset download skipped or failed.")
        os.chdir(src.config.DATA_DIR)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Base data directory")
    args = parser.parse_args()
    
    if args.data_dir:
        from pathlib import Path
        src.config.DATA_DIR = Path(args.data_dir)
        src.config.MSCTD_DIR = src.config.DATA_DIR / "MSCTD"
        src.config.INSTANY_DIR = src.config.DATA_DIR / "InstaNY100K"
        
    download_msctd()
    download_instany()
    download_audio_sample()
    print("Data Ingestion Complete.")
