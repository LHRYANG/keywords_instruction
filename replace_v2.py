


import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from V2.utils import load_txt_keywords, KeywordSampler, TypedKeywordSampler, create_attribute_samplers
from V2.utils import parse_args, sample_reference_text, batch_generate,save_results
from V2.prompt_builder import build_replace_prompt
import torch 


if __name__ == "__main__":
    args = parse_args()
    
    model_path = args.model_path

    noun_json_path = "inside_keywords/edit_type_add_noun_freq_by_length.json"
    attribute_folder = args.attribute_folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    object_sampler_insider = KeywordSampler(noun_json_path)###,123选一个

    samplers = create_attribute_samplers(attribute_folder, avoid_name="color") 
    print(f"Loaded {len(samplers)} attribute samplers from {attribute_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    
    total_samples = 0
    for kkk in range(0, args.rounds):
        for complexity in [1, 2, 3, 0]:
            all_results = []
            print(f"Generating motion keywords with complexity {complexity} for round {kkk}...")
            for i in range(30):
                #batch_objects = noun_list[i:i + batch_size]
                prob = random.random()
                if  prob < 0.9:
                    length = 1
                elif prob<1.1:
                    length = 2
                else:
                    length = 3
                sampled_object = object_sampler_insider.sample(length=length, k=args.batch_size, alpha=0.75)
                
                reference_text = sample_reference_text(samplers, k=5, kk=10)
                #print("reference_text:", reference_text)
                prompts = [build_replace_prompt(obj, reference_text,max_attributes=complexity) for obj in sampled_object]
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