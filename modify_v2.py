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
        ##不考虑color
        if "color" in json_file.stem:
            continue
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

    return (
        f"Given an object name, please generate up to 5 natural phrase pairs that each describe the same object "
        f"with a single attribute modified.\n\n"
        f"- Each phrase pair should refer to the **same object** but differ in **exactly one attribute**.\n"
        f"- The attribute must come from a **single expansion dimension**, such as color, material, state, size, position, or style.\n"
        f"- All phrase pairs for one object must use the **same dimension**, but the attribute values should differ in each pair.\n"
        f"- Use common, natural expressions, e.g., 'a red umbrella' and 'a blue umbrella'.\n\n"
        f"You may refer to the following attribute keywords:\n{reference_text}\n\n"
        f"Return the result as a list of phrase pairs and only return the list. Format:\n"
        f"[(\"description 1-1\", \"description 1-2\"), (\"description 2-1\", \"description 2-2\"), ...]\n\n"
        f"Please output the list results for the object: {object_name}, Output:\n"
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
    ### color单独拿出来
    attribute_folder = "./outside_keywords"
    output_dir = "./modify_generated_results"
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    noun_list = load_nouns_from_json(noun_json_path)
    
    for kkk in range(1,50):
        random.shuffle(noun_list)
        batch_size = 16
        for complexity in [1, 2, 3]:
            all_results = []
            for i in tqdm(range(0, len(noun_list), batch_size), desc=f"Generating (complexity={complexity})"):
                batch_objects = noun_list[i:i + batch_size]
                reference_text = sample_keywords_from_json(attribute_folder, max_per_category=5)
                prompts = [build_prompt(obj, reference_text, max_attributes=complexity) for obj in batch_objects]
                #print(prompts[0])
                batch_results = batch_generate(model, tokenizer, batch_objects, prompts)
                #print(batch_results)
                all_results.extend(batch_results)
                #if int(i/batch_size) == 5:
                    #break 
            # 保存结果
            save_path = os.path.join(output_dir, f"keyword_generation_complexity_{complexity}_{kkk}.json")
            save_results(all_results, save_path)
            print(f"✅ Saved {len(all_results)} results to {save_path}")
