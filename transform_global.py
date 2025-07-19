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

def load_scenes_from_json(json_path_list, max_length=3) -> List[str]:
    result = []
    #print("json_path_list: ",json_path_list)
    for json_path in json_path_list:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        #print(data)
        for k in data:
            length = int(k.split("_")[-1])
            if 1 <= length <= max_length:
                result.extend(list(data[k].keys()))
    #print("aaaaaaaaaa: ",result)        
    result = list(set(result))
    #print(len(result))
    sampled_words = random.sample(result, min(len(result), 50)) 
    #print("sampled_words: ",sampled_words)
    return ", ".join(sampled_words)

def sample_keywords_from_json(folder: str, max_per_category=10) -> str:
    all_jsons = list(Path(folder).glob("*.json"))
    lines = []
    for json_file in all_jsons:
        ##不考虑color
        # if "color" not in json_file.stem:
        #     continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sampled = {}
        for category, words in data.items():
            sampled_words = random.sample(words, min(len(words), max_per_category))
            sampled[category] = sampled_words
        line = f"{json_file.stem}: {json.dumps(sampled, ensure_ascii=False)}"
        lines.append(line)
    return "\n".join(lines)


# def sample_keywords_from_json_only_scene(folder: str, max_per_category=10) -> str:
#     all_jsons = list(Path(folder).glob("*.json"))
#     lines = []
#     for json_file in all_jsons:
#         ##不考虑color
#         if "color" in json_file.stem:
#             continue
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         sampled = {}
#         for category, words in data.items():
#             sampled_words = random.sample(words, min(len(words), max_per_category))
#             sampled[category] = sampled_words
#         line = f"{json_file.stem}: {json.dumps(sampled, ensure_ascii=False)}"
#         lines.append(line)
#     return "\n".join(lines)


def build_prompt(object_name: str, reference_text: str, reference_scene: str, max_attributes=1) -> str:
    # return (
    #     f"Given an object, please generate up to 5 slightly more complex but still natural keyword phrases.\n\n"
    #     f"Each phrase should include **{max_attributes}** additional attribute{'s' if max_attributes > 1 else ''}, "
    #     f"such as color, material, state, size, position, or style.\n\n"
    #     f"You may refer to the following attribute keywords:\n{reference_text}\n\n"
    #     f"Return the result as a list and only return the list. For example:\n"
    #     f"Given 'umbrella', Output:\n"
    #     f"{{\"umbrella\": [\"description 1\", \"description 2\", \"description 3\"]}}\n\n"
    #     f"Please output the JSON results for the object: {object_name}, Output:\n"
    # )

    # return (
    #     f"Given an object name, please generate up to 5 natural phrase pairs that each describe the same object "
    #     f"with a single attribute modified.\n\n"
    #     f"- Each phrase pair should refer to the **same object** but differ in **exactly one attribute**.\n"
    #     f"- The attribute must come from a **single expansion dimension**, such as color, material, state, size, position, or style.\n"
    #     f"- All phrase pairs for one object must use the **same dimension**, but the attribute values should differ in each pair.\n"
    #     f"- Use common, natural expressions, e.g., 'a red umbrella' and 'a blue umbrella'.\n\n"
    #     f"You may refer to the following attribute keywords:\n{reference_text}\n\n"
    #     f"Return the result as a list of phrase pairs and only return the list. Format:\n"
    #     f"[(\"description 1-1\", \"description 1-2\"), (\"description 2-1\", \"description 2-2\"), ...]\n\n"
    #     f"Please output the list results for the object: {object_name}, Output:\n"
    # )

    if max_attributes == 0:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **scene attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but in different scenes.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella in rainy day', 'a red umbrella in sunny day')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the scene in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- You may reference the following scenes:\n{reference_scene}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the scene-change phrase pairs for the object: {object_name}\nOutput:\n"
        )
    else:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **scene attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but in different scenes.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella in rainy day', 'a red umbrella in sunny day')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the scene in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- You may reference the following scenes:\n{reference_scene}\n\n"
            f"You can also add {max_attributes} other attributes, such as material, state, size, position, or style to describe the object or the scene. But the difference should only be in the scene attribute. Here are some reference values for other attributes:\n{reference_text}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the scene-change phrase pairs for the object: {object_name}\nOutput:\n"
        )
        


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
        #print("response: ",decoded)
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
            #     "error": f"Parsing failed: {str(e)}",
            #     "raw": decoded[:200]
            # })
            print(f"❌ Error parsing output for {item}: {e} {decoded}")
    return results

def save_results(results: List[Dict], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    model_path = "/scratch/dyvm6xra/share_data/Qwen2.5-7B-Instruct"
    noun_json_path = "inside_keywords/edit_type_add_noun_freq_by_length.json"
    ### scene单独拿出来
    attribute_folder = "./outside_keywords"
    scene_file_list = ["inside_keywords/edit_type_background_change_noun_freq_by_length.json","inside_keywords/edit_type_env_noun_freq_by_length.json", "inside_keywords/edit_type_tone_transfer_change_by_length.json","inside_keywords/edit_type_tune_transfer_change_by_length.json"]
    output_dir = "./change_global_generated_results"
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    noun_list = load_nouns_from_json(noun_json_path)
    
    for kkk in range(1,50):
        random.shuffle(noun_list)
        batch_size = 16
        for complexity in [0, 1, 2, 3]:
            all_results = []
            for i in tqdm(range(0, len(noun_list), batch_size), desc=f"Generating (complexity={complexity})"):
                batch_objects = noun_list[i:i + batch_size]
                reference_text = sample_keywords_from_json(attribute_folder, max_per_category=10)
                reference_scenes = load_scenes_from_json(scene_file_list,max_length=3)
                prompts = [build_prompt(obj, reference_text, reference_scenes, max_attributes=complexity) for obj in batch_objects]
                #print(prompts[0])
                batch_results = batch_generate(model, tokenizer, batch_objects, prompts)
                #print(batch_results)
                all_results.extend(batch_results)
                # if int(i/batch_size) == 5:
                #     break 
            # 保存结果
            save_path = os.path.join(output_dir, f"keyword_generation_complexity_{complexity}_{kkk}.json")
            save_results(all_results, save_path)
            print(f"✅ Saved {len(all_results)} results to {save_path}")
