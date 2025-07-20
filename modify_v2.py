import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from typing import List, Dict
from V2.utils import load_txt_keywords, KeywordSampler, TypedKeywordSampler, create_attribute_samplers2
from V2.utils import parse_args, sample_reference_text, batch_generate,save_results
from V2.prompt_builder import build_modify_change_prompt
import torch 

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

if __name__ == "__main__":
    args = parse_args()
    
    model_path = args.model_path

    noun_json_path = "inside_keywords/edit_type_add_noun_freq_by_length.json"
    attribute_folder = args.attribute_folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    object_sampler_insider = KeywordSampler(noun_json_path)###,123选一个

    samplers = create_attribute_samplers2(attribute_folder, use_name=["material","size","shape", "texture"])
    other_attributes_samplers = create_attribute_samplers2(attribute_folder, use_name=["action","color","spatial"])
    #print(f"Loaded {len(samplers)} attribute samplers from {attribute_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    #noun_list = load_nouns_from_json(noun_json_path)
    
    total_samples = 0
    for kkk in range(0, args.rounds):
        for complexity in [1, 2, 3, 0]:
            all_results = []
            print(f"Generating motion keywords with complexity {complexity} for round {kkk}...")
            for i in range(50):
                #batch_objects = noun_list[i:i + batch_size]
                prob = random.random()
                if  prob < 0.85:
                    length = 1
                elif prob<0.95:
                    length = 2
                else:
                    length = 3
                sampled_object = object_sampler_insider.sample(length=length, k=args.batch_size, alpha=0.75)
                #print(sampled_object)
                #sampled_color_1 = color_sampler_insider.sample(length=length, k=5, alpha=0.75)
                sampled_modified_attr = sample_reference_text(samplers, k=5, kk=5)
                sampled_attr = sample_reference_text(other_attributes_samplers, k=5, kk=5)
                
                prompts = [build_modify_change_prompt(obj, sampled_attr, sampled_modified_attr, max_attributes=complexity) for obj in sampled_object]
                #print(prompts[0])
                batch_results = batch_generate(model, tokenizer, sampled_object, prompts, outtype="list")
                #print("success: ",batch_results[0])
                all_results.extend(batch_results)
                
            # 保存结果
            save_path = os.path.join(output_dir, f"keyword_generation_complexity_{complexity}_{kkk}.json")
            save_results(all_results, save_path)
            print(f"✅ Saved {len(all_results)} results to {save_path}")
            total_samples += len(all_results)
    
    print(f"Total samples generated: {total_samples}")