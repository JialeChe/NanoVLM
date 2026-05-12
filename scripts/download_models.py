#!/usr/bin/env python3
"""
NanoVLM 模型权重下载脚本
下载 Vision Encoder (SigLIP) 和 Language Model (Qwen2) 的预训练权重

用法:
    python scripts/download_models.py                    # 下载全部
    python scripts/download_models.py --vision_only      # 只下载 Vision
    python scripts/download_models.py --language_only    # 只下载 Language
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODELS = {
    "vision": {
        "name": "SigLIP",
        "model_id": "google/siglip-so400m-patch14-384",
        "save_dir": "models/siglip-so400m-patch14-384",
        "description": "SigLIP Vision Encoder (400M parameters)",
    },
    "language": {
        "name": "Qwen2-0.5B",
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "save_dir": "models/Qwen2-0.5B-Instruct",
        "description": "Qwen2-0.5B Language Model (~0.5B parameters)",
    },
}


def download_from_hf(model_id: str, save_dir: str) -> bool:
    """使用 huggingface-cli 下载模型"""
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"  Model already exists at {save_dir}")
        print(f"  To re-download, delete the directory first.")
        return True

    try:
        import huggingface_hub
        from huggingface_hub import snapshot_download

        print(f"  Using huggingface_hub...")
        os.makedirs(save_dir, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  ✓ Downloaded to {save_dir}")
        return True

    except ImportError:
        # 尝试用 git-lfs + git clone
        try:
            print(f"  Using git clone (with LFS)...")
            cmd = [
                "git", "clone",
                f"https://huggingface.co/{model_id}",
                save_dir,
            ]
            subprocess.run(cmd, check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "0"})
            print(f"  ✓ Downloaded to {save_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Git clone failed: {e}")
            return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def download_with_transformers(model_id: str, save_dir: str) -> bool:
    """使用 transformers AutoModel 下载"""
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"  Model already exists at {save_dir}")
        return True

    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"  Downloading with transformers...")
        os.makedirs(save_dir, exist_ok=True)
        AutoModel.from_pretrained(model_id, cache_dir=save_dir)
        AutoTokenizer.from_pretrained(model_id, cache_dir=save_dir)
        print(f"  ✓ Downloaded")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NanoVLM pretrained weights")
    parser.add_argument(
        "--vision_only",
        action="store_true",
        help="Only download Vision Encoder",
    )
    parser.add_argument(
        "--language_only",
        action="store_true",
        help="Only download Language Model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NanoVLM Model Download")
    print("=" * 60)
    print(f"Target directory: ./models/")
    print("=" * 60)

    # 确保基础目录存在
    os.makedirs("models", exist_ok=True)

    # 确定要下载哪些模型
    to_download = []
    if args.vision_only:
        to_download = ["vision"]
    elif args.language_only:
        to_download = ["language"]
    else:
        to_download = ["vision", "language"]

    success = True
    for key in to_download:
        info = MODELS[key]
        print(f"\n[{info['name']}]")
        print(f"  Model: {info['model_id']}")
        print(f"  Description: {info['description']}")
        print(f"  Save to: {info['save_dir']}")

        if args.force and os.path.exists(info["save_dir"]):
            import shutil
            shutil.rmtree(info["save_dir"])
            print("  (Cleared existing files)")

        ok = download_from_hf(info["model_id"], info["save_dir"])
        if not ok:
            # 回退方案
            ok = download_with_transformers(info["model_id"], info["save_dir"])

        if not ok:
            print(f"  Failed to download {info['name']}. Please download manually.")
            print(f"  Visit: https://huggingface.co/{info['model_id']}")
            success = False

    # 总结
    print("\n" + "=" * 60)
    if success:
        print("✓ Download complete!")
        print("\nExpected directory structure:")
        print("  models/")
        for key in to_download:
            print(f"    {MODELS[key]['save_dir']}/")
        print("\nYou can now run training:")
        print("  python scripts/train.py --create_dummy")
    else:
        print("✗ Some downloads failed. See errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()