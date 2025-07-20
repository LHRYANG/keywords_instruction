import json
import random
from collections import defaultdict
from typing import List, Dict
import argparse
import os
from pathlib import Path
import torch 
import re
def parse_args():
    parser = argparse.ArgumentParser(description="Motion Phrase Generation with LLM")
    
    # 模型和数据路径
    parser.add_argument('--model_path', type=str, default="/scratch/dyvm6xra/share_data/Qwen2.5-7B-Instruct", help="Path to LLM model")
    parser.add_argument('--attribute_folder', type=str, default='outside_keywords')
    parser.add_argument('--output_dir', type=str, default='V2_results/motion_generated_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--complexity', type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=512)

    return parser.parse_args()



def save_results(results: List[Dict], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_txt_keywords(txt_path: str) -> list[str]:
    with open(txt_path, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]
    return keywords

class KeywordSampler:
    def __init__(self, json_path: str):
        self.keyword_pool: Dict[int, list[str]] = defaultdict(list)
        self.weights_pool: Dict[int, list[float]] = defaultdict(list)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, keywords in data.items():
            length = int(key.split('_')[-1])
            for word, freq in keywords.items():
                self.keyword_pool[length].append(word)
                self.weights_pool[length].append(freq)

    def sample(self, length: int = None, k: int = 1, alpha: float = 0.75) -> list[str]:
        """从指定长度的关键词中采样k个；若未指定长度则随机从所有长度中选择"""
        if length is None:
            length = random.choice(list(self.keyword_pool.keys()))
        keywords = self.keyword_pool[length]
        weights = self.weights_pool[length]

        #print(keywords, weights,k)
        if len(keywords) < k:
            keywords = self.keyword_pool[1]
            weights = self.weights_pool[1]
        adjusted_weights = [w**alpha for w in weights]
        return random.choices(keywords, weights=adjusted_weights, k=k)

    def _adjust_weights(self, weights, alpha=0.75):
        # 调整权重，使得采样更倾向于低频词
        return [w**alpha for w in weights]


class TypedKeywordSampler:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data: Dict[str, List[str]] = json.load(f)

    def sample_all(self, k: int = 2) -> List[str]:
        """
        从每个类型中采样 k 个关键词（不重复），返回格式如：
        ["posture_states: sitting, leaning", "motion_states: jumping, skipping"]
        """
        results = []
        for category, keywords in self.data.items():
            if not keywords:
                continue
            sample_size = min(k, len(keywords))
            sampled = random.sample(keywords, sample_size)
            line = f"{category}: {', '.join(sampled)}"
            results.append(line)
        random.shuffle(results)  # 打乱顺序
        return results

    def sample_one_type(self, k: int = 2) -> str:
        """
        从一个随机类型中采样 k 个关键词
        """
        category = random.choice(list(self.data.keys()))
        keywords = self.data[category]
        sample_size = min(k, len(keywords))
        sampled = random.sample(keywords, sample_size)
        return f"{category}: {', '.join(sampled)}"




def create_attribute_samplers(folder: str, avoid_name= None) -> list[TypedKeywordSampler]:
    samplers = []
    for file in Path(folder).glob("*.json"):
        if avoid_name is not None and file.stem == avoid_name:
            continue
        samplers.append(TypedKeywordSampler(str(file)))
    return samplers

def sample_reference_text(samplers: list, k: int = 3, kk: int = 5) -> str:
    """
    从每个 TypedKeywordSampler 中采样 k 个关键词，并拼接为多行文本。
    """
    lines = []
    #sampled_sampler = random.sample(samplers, kk)
    for sampler in samplers:
        lines.extend(sampler.sample_all(k))
    #random.shuffle(lines)  # 可选：打乱顺序更自然

    lines = random.sample(lines, min(len(lines), kk))  # 确保不超过 kk 行 
    return "\n".join(lines)



def create_attribute_samplers2(folder: str, use_name= None) -> list[TypedKeywordSampler]:
    samplers = []


    for file in Path(folder).glob("*.json"):
        use = False 
        for avd in use_name:
            if avd in file.stem:
                use = True 
        if not use:
            continue
        samplers.append(TypedKeywordSampler(str(file)))
    return samplers




def batch_generate(model, tokenizer, object_names: List[str], prompts: List[str], max_new_tokens=512, outtype="list", task=None) -> List[Dict]:
    
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
            )
        for prompt in prompts
    ]
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1,
            #pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for item, output_ids in zip(object_names, outputs):
        input_len = inputs.input_ids.shape[1]
        # 解码生成部分
        decoded = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True).strip()
        if outtype == "list":
            parsed = extract_phrase_pairs_from_text(decoded, item, task=task)
            results.append(parsed)
        elif outtype == "json":
            try:
                json_start = decoded.find('{')
                json_end = decoded.rfind('}') + 1
                json_str = decoded[json_start:json_end]

                response = json.loads(json_str)
                
                if isinstance(response, dict) and item in response and isinstance(response[item], list):
                    results.append({
                        "object": item,
                        "expanded_keywords": response[item]
                    })
            except Exception as e: 
                #print("error:", decoded)
                pass 

            
    return results




        


def is_english_text(s: str, ratio: float = 0.95) -> bool:
    """
    判断字符串是否主要为英文。
    """
    english_chars = re.findall(r'[a-zA-Z .,;:!?\"\'-]', s)
    return len(english_chars) / max(1, len(s)) >= ratio


def extract_phrase_pairs_from_text(decoded: str, object_name: str, task: str) -> dict | None:
    """
    从 LLM 输出中提取所有 (text1, text2) 英文句对。
    
    返回：
        dict 格式如下：
        {
            "object": object_name,
            "expanded_keywords": [("phrase 1", "phrase 2"), ...]
        }
        如果无有效对，返回 None。
    """
    # 提取所有 "(..., ...)" 形式的元组字符串
    tuple_pattern = r"\(([^()]+?,[^()]+?)\)"
    matches = re.findall(tuple_pattern, decoded)

    results = []
    for match in matches:
        try:
            part1, part2 = match.split(",", 1)
            phrase1 = part1.strip(" \"'")
            phrase2 = part2.strip(" \"'")
            if is_english_text(phrase1) and is_english_text(phrase2):
                if task =="global":
                    words1 = set(phrase1.split())
                    words2 = set(phrase2.split())
                    if len(words1.symmetric_difference(words2)) < 3:
                        results.append([phrase1, phrase2])
                    # else:
                    #     print(words1, words2)
                else:
                    results.append([phrase1, phrase2])
        except Exception:
            continue  # 忽略解析失败的 pair

    if results:
        return {
            "object": object_name,
            "expanded_keywords": results
        }
    else:
        return None