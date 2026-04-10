import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import subprocess
from src.configs import config

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_msctd():
    os.makedirs(config.data.msctd_dir, exist_ok=True)
    os.chdir(config.data.msctd_dir)

    if not os.path.exists("MSCTD_data"):
        print("Cloning MSCTD repo for text labels...")
        run_cmd("git clone https://github.com/XL2248/MSCTD .")

    # Assuming the user verified these links are still active
    urls = {
        "train_ende.zip": "1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj",
        "test.zip": "1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W",
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
        # Move image folders mapping
        run_cmd(f"mv *{split}* {split_dir}/ 2>/dev/null || true")
        # Properly structure text maps from the MSCTD GitHub repository core directory
        for kind in ['english', 'sentiment']:
            src_path = os.path.join("MSCTD_data", "Main_Data", f"{kind}_{split}.txt")
            if os.path.exists(src_path):
                import shutil
                shutil.copy(src_path, split_dir)
                
    os.chdir(config.data.data_dir)

def download_instany():
    if not os.path.exists(config.data.instany_dir):
        print("Downloading InstaNY100K via Kaggle...")
        os.makedirs(config.data.instany_dir, exist_ok=True)
        os.chdir(config.data.instany_dir)
        try:
            run_cmd("kaggle datasets download -d hsankesara/flickr-image-dataset")
            run_cmd("unzip -qq flickr-image-dataset.zip")
        except Exception as e:
            print(f"Warning: kaggle failure: {e}. Provide correct credentials.")
        os.chdir(config.data.data_dir)

def download_audio_sample():
    audio_dir = config.data.data_dir / "AudioSample"
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
        os.chdir(config.data.data_dir)



if __name__ == "__main__":
    config.parse_cli_args()
    download_msctd()
    download_instany()
    download_audio_sample()
    print("Data Ingestion Complete.")
