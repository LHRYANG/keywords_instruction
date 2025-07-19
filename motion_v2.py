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
from V2.prompt_builder import build_motion_prompt
import torch 






if __name__ == "__main__":
    
    args = parse_args()
    
    model_path = args.model_path
    ### fixed
    animal_txt = "outside_keywords/animal.txt"
    person_txt = "outside_keywords/person.txt"

    animal_list = load_txt_keywords(animal_txt)
    person_list = load_txt_keywords(person_txt)
    print(f"Loaded {len(animal_list)} animals and {len(person_list)} persons.")
    motion_file1 = "inside_keywords/edit_type_action_change_change_by_length.json"
    motion_file2 = "outside_keywords/keyword_action.json"
    
    motion_sampler_insider = KeywordSampler(motion_file1)
    #sampled_motion_1 = motion_sampler_insider.sample(length=1, k=5, alpha=0.25)  # 从1-gram中采样5个 ## 1-3随机选一个？
    #print(sampled_motion_1)
    # 2. 使用 TypedKeywordSampler 加载多类无频率关键词
    motion_sampler_outsider = TypedKeywordSampler(motion_file2)
    #sampled_motion_2 = motion_sampler_outsider.sample_all(k=3)       # 每类采3个关键词
    #sampled_motion_3 = motion_sampler_outsider.sample_one_type(k=5)  # 从某个类采5个

    attribute_folder = args.attribute_folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    samplers = create_attribute_samplers(attribute_folder, avoid_name="action")
    print(f"Loaded {len(samplers)} attribute samplers from {attribute_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    total_samples = 0  
    for kkk in range(0, args.rounds):
        for complexity in [1, 2, 3, 0]:
            random.shuffle(animal_list)
            print(f"Generating motion keywords with complexity {complexity} for round {kkk}...")
            all_results = []
            for i in range(0, len(animal_list), args.batch_size):
                batch_animal = animal_list[i:i + args.batch_size]
                batch_person = random.sample(person_list, 5)
                batch_objects = batch_animal + batch_person

                prob = random.random()
                if  prob < 0.7:
                    length = 1
                elif prob<0.9:
                    length = 2
                else:
                    length = 3
                sampled_motion_1 = motion_sampler_insider.sample(length=length, k=5, alpha=0.25)
                sampled_motion_2 = motion_sampler_outsider.sample_all(k=3)

                merged = set(sampled_motion_1)
                for line in sampled_motion_2:
                    if ':' in line:
                        _, keywords_str = line.split(':', 1)
                        keywords = [kw.strip() for kw in keywords_str.split(',')]
                        merged.update(keywords)
                sampled_motion = list(merged)

                #print(sampled_motion_1)
                #print(sampled_motion_2)
                reference_text = sample_reference_text(samplers, k=5, kk=10)
                #print("aaaaaaaa:",reference_text)

                prompts = [build_motion_prompt(obj, ", ".join(sampled_motion), reference_text, max_attributes=complexity) for obj in batch_objects]
                #print(prompts[0])
                batch_results = batch_generate(model, tokenizer, batch_objects, prompts)
                #print("success: ",batch_results[0])
                all_results.extend(batch_results)
            save_path = os.path.join(args.output_dir, f"keyword_generation_complexity_{complexity}_{kkk}.json")
            save_results(all_results, save_path)
            print(f"✅ Saved {len(all_results)} results to {save_path}")
            total_samples+=len(all_results)
    print(f"Total samples generated: {total_samples}")