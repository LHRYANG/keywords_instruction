import os
import json
import re
import multiprocessing
from pathlib import Path
from PIL import Image
import webdataset as wds
from tqdm import tqdm

def safe_decode_image_multi(sample):
    image_list = []
    image_indices = []
    image_name_list = []
    for k in list(sample.keys()):
        if k.endswith(".jpg") or k.endswith(".jpeg") or k.endswith(".png"):
            try:
                image = sample[k]
                image_list.append(image)
                image_name_list.append(k)
                image_indices.append(int(k.split(".")[0]))
            except Exception as e:
                print(f"⚠️ Error decoding {k} in sample {sample.get('__key__', '')}: {e}")

    if len(image_list) == 0:
        return None

    image_list_sorted = [x for _, x in sorted(zip(image_indices, image_list))]
    sample["image_list"] = image_list_sorted
    sample["image_indices"] = sorted(image_indices)
    return sample

def filter_none(sample):
    return sample is not None

def is_english(text, threshold=0.9):
    """判断是否为英文文本"""
    if not text:
        return False
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;:!?\"'-"
    english_chars = re.findall(f"[{re.escape(allowed_chars)}]", text)
    return len(english_chars) / max(1, len(text)) >= threshold

def process_tar_file(tar_path):
    tar_path = str(tar_path)
    is_v1_not_used = "v1_not_used" in tar_path

    total_tokens = 0
    total_samples = 0
    empty_caption0 = 0
    empty_caption1 = 0
    empty_change0 = 0
    empty_change1 = 0
    non_english_caption0 = 0
    non_english_caption1 = 0
    non_english_change0 = 0
    non_english_change1 = 0
    try:
        dataset = (
            wds.WebDataset(tar_path)
            .decode("pil")
            .rename(json="json")
            .map(safe_decode_image_multi)
            .select(filter_none)
        )

        for sample in dataset:
            img_list = sample["image_list"]
            images = img_list if is_v1_not_used else img_list[:2]  # v1: 所有图, 其他: 前两张图
            if len(img_list)>2 and not is_v1_not_used:
                continue 
            for img in images:
                if isinstance(img, Image.Image):
                    w, h = img.size
                    total_tokens += (w * h) / (14 * 14)

            if not is_v1_not_used:
                js = sample.get("json", {})
                if "qwen_0_caption" not in js:
                    continue 
                caption0 = js.get("qwen_0_caption", "")
                caption1 = js.get("qwen_0_caption", "")
                change0 = js.get("qwen_0_1_change", "")
                change1 = js.get("qwen_1_0_change", "")
                if not caption0.strip():
                    empty_caption0 += 1
                if not caption1.strip():
                    empty_caption1 += 1
                if not change0.strip():
                    empty_change0 += 1
                if not change1.strip():
                    empty_change1 += 1
                
                if not is_english(caption0):
                    non_english_caption0 += 1
                if not is_english(caption1):
                    non_english_caption1 += 1
                if not is_english(change0):
                    non_english_change0 += 1
                if not is_english(change1):
                    non_english_change1 += 1

            total_samples += 1

    except Exception as e:
        print(f"[ERROR] {tar_path}: {e}")

    return (total_samples, total_tokens, empty_caption0, empty_caption1, empty_change0, empty_change1,
            non_english_caption0, non_english_caption1, non_english_change0, non_english_change1)

def find_all_tar_files(root_dir):
    return list(Path(root_dir).rglob("*.tar"))

def main(root_dir, num_workers=8):
    tar_files = find_all_tar_files(root_dir)
    print(f"Found {len(tar_files)} tar files.")

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_tar_file, tar_files), total=len(tar_files)))

    total_samples = sum(x[0] for x in results)
    total_tokens = sum(x[1] for x in results)
    empty_caption0 = sum(x[2] for x in results)
    empty_caption1 = sum(x[3] for x in results)
    empty_change0 = sum(x[4] for x in results)
    empty_change1 = sum(x[5] for x in results)
    non_english_caption0 = sum(x[6] for x in results)
    non_english_caption1 = sum(x[7] for x in results)
    non_english_change0 = sum(x[8] for x in results)
    non_english_change1 = sum(x[9] for x in results)

    print("\n=== Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"Total image tokens: {total_tokens:.2f}")
    if total_samples > 0:
        print(f"Avg tokens per sample: {total_tokens / total_samples:.2f}")

    print("\n=== Empty Content Statistics ===")
    print(f"Empty caption0: {empty_caption0}")
    print(f"Empty caption1: {empty_caption1}")
    print(f"Empty change0: {empty_change0}")
    print(f"Empty change1: {empty_change1}")

    print("\n=== Non-English Content Statistics ===")
    print(f"Non-English caption0: {non_english_caption0}")
    print(f"Non-English caption1: {non_english_caption1}")
    print(f"Non-English change0: {non_english_change0}")
    print(f"Non-English change1: {non_english_change1}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory containing tar files")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args.root, args.num_workers)
