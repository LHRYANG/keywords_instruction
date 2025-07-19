# import json
# import os
# import json
# import random
# from transformers import AutoModelForCausalLM, AutoTokenizer

# def load_nouns_from_json(json_path, lengths=(1, 2)):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     selected_nouns = set()

#     for length in lengths:
#         key = f"keyword_length_{length}"
#         if key in data:
#             selected_nouns.update(data[key].keys())

#     return selected_nouns


# def sample_keywords_from_json(json_dir, max_per_category=5):
#     lines = []

#     for fname in sorted(os.listdir(json_dir)):
#         if not fname.endswith(".json"):
#             continue

#         path = os.path.join(json_dir, fname)
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#         except Exception as e:
#             print(f"❌ Failed to read {fname}: {e}")
#             continue

#         parts = []
#         for category, items in data.items():
#             if isinstance(items, list) and items:
#                 sampled = random.sample(items, min(len(items), max_per_category))
#                 parts.append(f"{category}: {sampled}")
#         line = f"{fname}: " + "; ".join(parts)
#         line = line.replace(".json",'')
#         lines.append(line)

#     return "\n".join(lines)


# # 读取关键词列表
# def read_keywords(txt_path):
#     with open(txt_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#     return [line.strip() for line in lines if line.strip()]


# # 构造 prompt
# def build_prompt(object_name):
#     return (
#         f"请基于'{object_name}'生成最多5个稍复杂但仍然自然的关键词短语。"
#         f"每个短语只能添加一个属性，例如颜色、材质、状态、大小、位置、风格等。"
#         f"返回格式为一个 JSON 列表。例如：['red umbrella', 'folded umbrella', ...]"
#     )

# # 主函数：生成所有 prompt 并保存
# def build_prompt(object_name, reference_text, max_attributes=1):
#     """构建英文 prompt，限制属性数量"""
#     return (
#     f"Given an object name, generate up to 5 slightly more complex but still natural keyword phrases.\n\n"
#     f"Each phrase should include {max_attributes} additional attribute{'s' if max_attributes > 1 else ''}, "
#     f"such as color, material, state, size, position, or style.\n\n"
#     f"You may refer to the following attribute examples:\n{reference_text}\n\n"
#     f"Return the result as a JSON dictionary and output **only** the JSON. For example, if the object is 'umbrella', return:\n"
#     f"{{\"umbrella\": [\"red umbrella\", \"folded umbrella\", \"transparent umbrella\"]}}"
# )


# BATCH_SIZE = 16

# # 示例用法
# if __name__ == "__main__":
#     model_path = "/scratch/dyvm6xra/share_data/Qwen2.5-7B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.padding_side = 'left'
    
#     noun_list = load_nouns_from_json("inside_keywords/edit_type_add_noun_freq_by_length.json")
#     #print(noun_list[0:10])
#     for complexity in [1,2,3]:
#         for cur_object in noun_list:
#             reference_text = sample_keywords_from_json("./outside_keywords", max_per_category=5)
#             prompt = build_prompt(cur_object,reference_text, max_attributes=complexity)+"\n\nPlease output the json results for the #object: "+cur_object


import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


def load_nouns_from_json(json_path: str, max_length=1) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for k in data:
        length = int(k.split("_")[-1])
        if 1 <= length <= max_length:
            result.extend(list(data[k].keys()))
    return result


def sample_keywords_from_json(folder: str, max_per_category=5) -> str:
    all_jsons = list(Path(folder).glob("*.json"))
    lines = []
    for json_file in all_jsons:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sampled = {}
        for category, words in data.items():
            sampled_words = random.sample(words, min(len(words), max_per_category))
            sampled[category] = sampled_words
        line = f"{json_file.stem}: {json.dumps(sampled, ensure_ascii=False)}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(object_name: str, reference_text: str, max_attributes=1) -> str:

    if max_attributes==0:
        return f"""
You are given an object name and asked to generate a list of **natural and plausible phrase pairs**, where each pair describes two objects that can reasonably **replace each other** in a scene.

- Each phrase should be short and descriptive, suitable for use in visual editing prompts or image descriptions.
- Replacements can vary in **type, category, function, or style** — they should **not** have the same identity or semantic role.
- Use **common or creative substitutions**, as long as they are visually plausible in some context.

Return only a list of 5 phrase pairs in this format:
[
("phrase1-1", "phrase1-2"),
("phrase2-1", "phrase2-2"),
...
]

Please output the results for the object: {object_name}
Output:\n 
"""
    else:
        return f"""
You are given an object name and asked to generate a list of **natural and plausible phrase pairs**, where each pair describes two objects that can reasonably **replace each other** in a scene.

- Each phrase should be short and descriptive, suitable for use in visual editing prompts or image descriptions.
- Replacements can vary in **type, category, function, or style** — they should **not** have the same identity.
- Use **common or creative substitutions**, as long as they are visually plausible in some context. For example:
(a red apple, a watermelon)
- Each phrase may contain **up to {max_attributes} attribute{'s' if max_attributes > 1 else ''}**, such as color, material, shape, size, or function.

Reference attribute keywords for inspiration:
{reference_text}

Return only a list of 5 phrase pairs in this format:
[
("phrase1-1", "phrase1-2"),
("phrase2-1", "phrase2-2"),
...
]

Please output the results for the object: {object_name}
Output:\n 
"""
def batch_generate(model, tokenizer, object_names: List[str], prompts: List[str], max_new_tokens=256) -> List[Dict]:
    
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
            )
        for prompt in prompts
    ]
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    #inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.5,
            pad_token_id=tokenizer.eos_token_id,
        )
    #decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    for item, output_ids in zip(object_names, outputs):
        input_len = inputs.input_ids.shape[1]
        # 解码生成部分
        decoded = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True).strip()

        # 尝试解析 JSON
        try:
            list_start = decoded.find('[')
            list_end = decoded.rfind(']') + 1
            list_str = decoded[list_start:list_end]

            # 尝试用 eval 加载 tuple list（注意安全风险，只对信任的模型输出使用）
            expanded_list = eval(list_str)

            if isinstance(expanded_list, list) and all(
                isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in expanded_list
            ):
                results.append({
                    "object": item,
                    "expanded_keywords": [tuple(pair) for pair in expanded_list]
                })
            # else:
            #     results.append({
            #         "object": item,
            #         "expanded_keywords": [],
            #         "error": "Invalid format"
            #     })

        except Exception as e:
            # results.append({
            #     "object": item,
            #     "expanded_keywords": [],
            #     "error": f"JSON parsing failed: {str(e)}",
            #     "raw": decoded[:200]  # 可选：保留部分原始输出做调试
            # })
            print(f"❌ Error parsing output for {item}: {e} {decoded}")
    return results

def save_results(results: List[Dict], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    model_path = "/scratch/dyvm6xra/share_data/Qwen2.5-7B-Instruct"
    noun_json_path = "inside_keywords/edit_type_add_noun_freq_by_length.json"
    attribute_folder = "./outside_keywords"
    output_dir = "./replace_generated_results"
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    noun_list = load_nouns_from_json(noun_json_path)
    
    for kk in range(1, 30):
        random.shuffle(noun_list)
        batch_size = 16
        for complexity in [0, 1, 2, 3]:
            all_results = []
            for i in tqdm(range(0, len(noun_list), batch_size), desc=f"Generating (complexity={complexity})"):
                batch_objects = noun_list[i:i + batch_size]
                reference_text = sample_keywords_from_json(attribute_folder, max_per_category=5)
                prompts = [build_prompt(obj, reference_text, max_attributes=complexity) for obj in batch_objects]
                batch_results = batch_generate(model, tokenizer, batch_objects, prompts)
                #print(batch_results)
                #print(batch_results)
                #print(prompts[0])
                all_results.extend(batch_results)
                # if int(i/batch_size) == 5:
                #     break 
            # 保存结果
            save_path = os.path.join(output_dir, f"keyword_generation_complexity_{complexity}_{kk}.json")
            save_results(all_results, save_path)
            print(f"✅ Saved {len(all_results)} results to {save_path}")
